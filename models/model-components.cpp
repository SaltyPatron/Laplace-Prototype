// Connected FFN decomposition over held SafeTensors source handles.
// The managed caller supplies the production recognizer's bindings. No tokenizer,
// tensor-name convention, persistent operator records or vocabulary pair scan.
#include <laplace/dynamics/model_math.h>
#include <laplace/dynamics/bilinear_edges.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <immintrin.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

struct TensorView {
    int32_t fd, dtype; // absent=-1; F32=0, F16=1, BF16=2
    uint64_t offset, rows, cols;
    int32_t transpose, reserved;
};
struct FfnBinding {
    TensorView embedding, up, gate, down, up_bias, gate_bias, down_bias;
    int32_t activation, layer;
};
struct ComponentReceipt {
    uint64_t units, decoded_values, scratch_bytes;
    double max_absolute_error, relative_error;
};
struct AttentionBinding {
    TensorView embedding, q, k, v, o, qb, kb, vb, ob, norm, norm_bias;
    int32_t heads, kv_heads, head_dim, rotary_dim, norm_kind, layer;
    double epsilon, rope_base;
};
struct AttentionReceipt {
    uint64_t heads, kv_heads, decoded_values;
    double projection_error, output_error, prefix_error, position_delta, context_delta;
};
static_assert(sizeof(AttentionBinding)==480 && sizeof(AttentionReceipt)==64);
static_assert(sizeof(TensorView)==40 && sizeof(FfnBinding)==288 && sizeof(ComponentReceipt)==40);

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Mapping {
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    explicit Mapping(int fd) {
        struct stat s{};
        require(fstat(fd, &s) == 0 && s.st_size > 0, "source handle unavailable");
        bytes = size_t(s.st_size);
        auto p = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        require(p != MAP_FAILED, "source mmap failed");
        data = static_cast<const uint8_t*>(p);
    }
    ~Mapping() { if (data) munmap(const_cast<uint8_t*>(data), bytes); }
    Mapping(const Mapping&) = delete;
};
struct Reader {
    const TensorView& view;
    const uint8_t* data;
    size_t rows, cols, width;
    Reader(const TensorView& v, std::map<int, Mapping>& maps) : view(v) {
        require(v.fd >= 0 && v.dtype >= 0 && v.dtype <= 2, "invalid source view");
        auto [at, fresh] = maps.try_emplace(v.fd, v.fd);
        const auto& m = at->second;
        width = v.dtype == 0 ? 4 : 2;
        require(v.rows && v.cols && v.rows <= SIZE_MAX / v.cols / width,
                "invalid source extent");
        require(v.offset <= m.bytes && v.rows*v.cols*width <= m.bytes-v.offset,
                "source view exceeds held file");
        data = m.data + v.offset;
        rows = v.transpose ? v.cols : v.rows;
        cols = v.transpose ? v.rows : v.cols;
    }
    double at(size_t row, size_t col) const {
        require(row < rows && col < cols, "source axis out of bounds");
        size_t i = view.transpose ? col*view.cols + row : row*view.cols + col;
        float value;
        if (view.dtype == 0) std::memcpy(&value, data + i*4, 4);
        else {
            uint16_t bits; std::memcpy(&bits, data + i*2, 2);
            value = view.dtype == 1 ? _cvtsh_ss(bits)
                                   : std::bit_cast<float>(uint32_t(bits) << 16);
        }
        require(std::isfinite(value), "nonfinite component operand");
        return value;
    }
    std::vector<float> matrix() const {
        std::vector<float> result(rows*cols);
        for (size_t r=0; r<rows; ++r)
            for (size_t c=0; c<cols; ++c) result[r*cols+c] = float(at(r,c));
        return result;
    }
};
double activate(double x, int kind) {
    switch (kind) {
    case 0: case 5: return x >= 0 ? x/(1+std::exp(-x)) : x*std::exp(x)/(1+std::exp(x));
    case 1: return 0.5*x*(1+std::erf(x/std::sqrt(2.0)));
    case 2: return 0.5*x*(1+std::tanh(std::sqrt(2.0/std::acos(-1.0))*(x+0.044715*x*x*x)));
    case 3: return x/(1+std::exp(-1.702*x));
    case 4: return std::max(0.0,x);
    default: throw std::runtime_error("unknown recognized activation");
    }
}
}

namespace {
std::vector<float> optional_vector(const TensorView& v,size_t n,std::map<int,Mapping>& maps) {
    if(v.fd<0)return {};
    Reader r(v,maps);require(r.rows*r.cols==n,"attention vector axis mismatch");return r.matrix();
}
std::vector<double> projection(const std::vector<double>& x,size_t n,size_t d,
    const std::vector<float>& w,const std::vector<float>& bias,size_t out) {
    std::vector<double> result(n*out);
    require(project_embedding_d(x.data(),n,d,w.data(),out,result.data())==0,"native attention projection failed");
    if(!bias.empty())require(add_row_vector_d(result.data(),n,out,bias.data())==0,"native attention bias failed");
    return result;
}
double relative_error(const std::vector<double>& a,const std::vector<double>& b) {
    require(a.size()==b.size(),"comparison cardinality");double error=0,norm=0;
    for(size_t i=0;i<a.size();i++){require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite attention result");
        double e=a[i]-b[i];error+=e*e;norm+=b[i]*b[i];}
    return norm>0?std::sqrt(error/norm):std::sqrt(error);
}
// HF source convention: the first rotary_dim/2 coordinates pair with the
// following rotary_dim/2; remaining head coordinates pass through unchanged.
void rotate(std::vector<double>& x,size_t n,size_t heads,size_t hd,size_t rd,double base,size_t gap) {
    for(size_t t=0;t<n;t++)for(size_t h=0;h<heads;h++)for(size_t j=0;j<rd/2;j++){
        double angle=double(t*gap)*std::pow(base,-2.0*double(j)/double(rd));
        size_t a=(t*heads+h)*hd+j,b=a+rd/2;double u=x[a],v=x[b];
        x[a]=u*std::cos(angle)-v*std::sin(angle);x[b]=v*std::cos(angle)+u*std::sin(angle);
    }
}
std::vector<double> attend(const std::vector<double>& q,const std::vector<double>& k,
    const std::vector<double>& v,size_t n,size_t heads,size_t kv,size_t hd) {
    std::vector<double> joined(n*heads*hd,0),scores(n);
    for(size_t t=0;t<n;t++)for(size_t h=0;h<heads;h++){
        size_t kh=h/(heads/kv);double maximum=-INFINITY;
        // Causal context: only this position and its predecessors participate.
        for(size_t s=0;s<=t;s++){double score=0;
            for(size_t c=0;c<hd;c++)score+=q[(t*heads+h)*hd+c]*k[(s*kv+kh)*hd+c];
            scores[s]=score/std::sqrt(double(hd));maximum=std::max(maximum,scores[s]);}
        double mass=0;for(size_t s=0;s<=t;s++){scores[s]=std::exp(scores[s]-maximum);mass+=scores[s];}
        for(size_t s=0;s<=t;s++)for(size_t c=0;c<hd;c++)
            joined[(t*heads+h)*hd+c]+=(scores[s]/mass)*v[(s*kv+kh)*hd+c];
    }
    return joined;
}
}

// Source-bound, independent attention blocks. Returned values include the
// residual input; each head's O-column contribution is checked against the
// whole native projection of the concatenated attended values.
extern "C" int lp_model_attention_components(const AttentionBinding* layers,size_t count,
    const uint64_t* token_rows,size_t n,double* results,size_t result_count,
    AttentionReceipt* receipts,char* error,size_t error_bytes) {
    try {
        require(layers&&count&&token_rows&&n>=2&&results&&receipts,"attention requires an ordered context of at least two bindings");
        std::map<int,Mapping> maps;size_t result_at=0;
        for(size_t layer=0;layer<count;layer++){
            const auto& b=layers[layer];Reader e(b.embedding,maps),qr(b.q,maps),kr(b.k,maps),vr(b.v,maps),out(b.o,maps);
            require(b.heads>0&&b.kv_heads>0&&b.head_dim>0&&b.rotary_dim>=0,"positive attention dimensions required");
            size_t d=e.cols,h=b.heads,kv=b.kv_heads,hd=b.head_dim,rd=b.rotary_dim;
            require(h<=SIZE_MAX/hd&&h%kv==0&&rd<=hd&&rd%2==0&&std::isfinite(b.rope_base)&&b.rope_base>0&&std::isfinite(b.epsilon)&&b.epsilon>0,"attention layout/position contract");
            size_t ad=h*hd,kd=kv*hd;require(ad==d,"attention head width must join the residual width");
            require(qr.rows==ad&&qr.cols==d&&kr.rows==kd&&kr.cols==d&&vr.rows==kd&&vr.cols==d&&out.rows==d&&out.cols==ad,"attention projection axes do not join");
            require(d<=SIZE_MAX/n&&result_at<=result_count&&n*d<=result_count-result_at,"attention result capacity");
            auto qw=qr.matrix(),kw=kr.matrix(),vw=vr.matrix(),ow=out.matrix();
            auto qb=optional_vector(b.qb,ad,maps),kb=optional_vector(b.kb,kd,maps),vb=optional_vector(b.vb,kd,maps),ob=optional_vector(b.ob,d,maps);
            auto gain=optional_vector(b.norm,d,maps),offset=optional_vector(b.norm_bias,d,maps);
            require(!gain.empty(),"recognized attention input normalization required");
            std::vector<double> residual(n*d),x(n*d);
            for(size_t t=0;t<n;t++)for(size_t c=0;c<d;c++)residual[t*d+c]=e.at(token_rows[t],c);
            x=residual;
            if(b.norm_kind==1)require(layer_norm_rows_d(x.data(),n,d,gain.data(),offset.empty()?nullptr:offset.data(),b.epsilon)==0,"native LayerNorm failed");
            else {require(b.norm_kind==0&&offset.empty(),"RMS normalization contract");
                for(size_t t=0;t<n;t++){double ss=0;for(size_t c=0;c<d;c++)ss+=x[t*d+c]*x[t*d+c];
                    double scale=1/std::sqrt(ss/d+b.epsilon);for(size_t c=0;c<d;c++)x[t*d+c]*=scale*gain[c];}}
            auto q=projection(x,n,d,qw,qb,ad),k=projection(x,n,d,kw,kb,kd),v=projection(x,n,d,vw,vb,kd);
            // Independently recover the Q/K/V rows for each query/shared KV head.
            double projection_error=0;
            auto check=[&](const std::vector<float>& w,const std::vector<float>& bias,const std::vector<double>& expected,size_t width){
                std::vector<double> direct(n*width);
                for(size_t t=0;t<n;t++)for(size_t r=0;r<width;r++){
                    double sum=bias.empty()?0:bias[r];for(size_t c=0;c<d;c++)sum+=x[t*d+c]*w[r*d+c];direct[t*width+r]=sum;}
                projection_error=std::max(projection_error,relative_error(direct,expected));};
            check(qw,qb,q,ad);check(kw,kb,k,kd);check(vw,vb,v,kd);
            auto q_unrotated=q,k_unrotated=k;
            rotate(q,n,h,hd,rd,b.rope_base,1);rotate(k,n,kv,hd,rd,b.rope_base,1);
            auto joined=attend(q,k,v,n,h,kv,hd);
            auto expected=projection(joined,n,ad,ow,ob,d);
            std::vector<double> sum(n*d,0);
            for(size_t t=0;t<n;t++)for(size_t head=0;head<h;head++)for(size_t c=0;c<d;c++){
                double contribution=0;for(size_t j=0;j<hd;j++)contribution+=joined[(t*h+head)*hd+j]*ow[c*ad+head*hd+j];
                sum[t*d+c]+=contribution;}
            for(size_t t=0;t<n;t++)for(size_t c=0;c<d;c++)if(!ob.empty())sum[t*d+c]+=ob[c];
            double output_error=relative_error(sum,expected);
            // Prefix invariance checks that future entries cannot alter position 0.
            auto prefix=attend(q,k,v,1,h,kv,hd);
            auto prefix_out=projection(prefix,1,ad,ow,ob,d);
            std::vector<double> first(expected.begin(),expected.begin()+d);
            double prefix_error=relative_error(first,prefix_out);
            // Moving the same entities to different relative positions exercises
            // RoPE as a live dependency, including Phi's unrotated complement.
            rotate(q_unrotated,n,h,hd,rd,b.rope_base,2);rotate(k_unrotated,n,kv,hd,rd,b.rope_base,2);
            auto moved=projection(attend(q_unrotated,k_unrotated,v,n,h,kv,hd),n,ad,ow,ob,d);
            // Replacing only the last value changes context without changing IDs
            // of the prefix; this result is a numerical check, not testimony.
            auto other_v=v;std::copy_n(v.data(),kd,other_v.data()+(n-1)*kd);
            auto changed=projection(attend(q,k,other_v,n,h,kv,hd),n,ad,ow,ob,d);
            receipts[layer]={h,kv,qw.size()+kw.size()+vw.size()+ow.size()+qb.size()+kb.size()+vb.size()+ob.size()+gain.size()+offset.size(),
                projection_error,output_error,prefix_error,relative_error(moved,expected),relative_error(changed,expected)};
            require(projection_error<1e-10&&output_error<1e-10&&prefix_error<1e-10,"attention decomposition or causal prefix check failed");
            for(size_t i=0;i<n*d;i++)results[result_at+i]=residual[i]+sum[i];
            result_at+=n*d;
        }
        require(result_at==result_count,"attention result shape");return 0;
    }catch(const std::exception& e){if(error&&error_bytes)std::snprintf(error,error_bytes,"%s",e.what());return -1;}
}

// One native call per model. Every FFN unit participates, including zero and
// negative effects. The component calculation uses strided views; the independent
// production FFN kernel supplies a numerical comparison, one layer at a time.
extern "C" int lp_model_ffn_components(const FfnBinding* layers, size_t count,
    uint64_t token_row, double* results, size_t result_count,
    ComponentReceipt* receipts, char* error, size_t error_bytes) {
    try {
        require(layers && count && receipts && results, "missing component bindings or result buffer");
        std::map<int, Mapping> maps;
        size_t result_at=0;
        for (size_t layer=0; layer<count; ++layer) {
            const auto& b = layers[layer];
            Reader embedding(b.embedding,maps), up(b.up,maps), down(b.down,maps);
            const size_t d=embedding.cols, units=up.rows;
            require(result_at<=result_count && d<=result_count-result_at,"result buffer capacity");
            require(up.cols==d && down.rows==d && down.cols==units,
                    "recognized FFN axes do not join");
            std::vector<double> input(d), output(d,0), reference(d);
            for (size_t k=0; k<d; ++k) input[k]=embedding.at(token_row,k);
            auto u=up.matrix(), w=down.matrix();
            std::vector<float> g, ub, gb, db;
            if (b.gate.fd>=0) {
                Reader gate(b.gate,maps);
                require(gate.rows==units && gate.cols==d,"gate axis mismatch");
                g=gate.matrix();
            }
            auto bias=[&](const TensorView& v,size_t length) {
                if (v.fd<0) return std::vector<float>{};
                Reader r(v,maps); require(r.rows*r.cols==length,"bias axis mismatch");
                return r.matrix();
            };
            ub=bias(b.up_bias,units); gb=bias(b.gate_bias,units); db=bias(b.down_bias,d);
            // Source tuple j is gate row j + up row j + down column j.
            for (size_t j=0; j<units; ++j) {
                double a=ub.empty()?0:ub[j], gate=gb.empty()?0:gb[j];
                for (size_t k=0; k<d; ++k) {
                    a+=input[k]*u[j*d+k];
                    if (!g.empty()) gate+=input[k]*g[j*d+k];
                }
                const double signal=g.empty()?activate(a,b.activation):activate(gate,b.activation)*a;
                for (size_t k=0; k<d; ++k) output[k]+=signal*w[k*units+j];
            }
            for (size_t k=0; k<d; ++k) if (!db.empty()) output[k]+=db[k];
            int rc=ffn_write_vectors_ex_d(input.data(),1,d,u.data(),ub.empty()?nullptr:ub.data(),
                g.empty()?nullptr:g.data(),gb.empty()?nullptr:gb.data(),units,w.data(),
                db.empty()?nullptr:db.data(),d,b.activation,reference.data());
            require(rc==0,"production FFN comparison rejected its bindings");
            double norm=0, delta=0, maximum=0;
            for (size_t k=0; k<d; ++k) {
                require(std::isfinite(output[k]) && std::isfinite(reference[k]),"nonfinite component result");
                double e=output[k]-reference[k]; delta+=e*e; norm+=reference[k]*reference[k];
                maximum=std::max(maximum,std::abs(e));
            }
            receipts[layer]={units,u.size()+g.size()+w.size()+ub.size()+gb.size()+db.size(),
                4*(u.size()+g.size()+w.size()+ub.size()+gb.size()+db.size())+(3*d+64*units)*sizeof(double),
                maximum,norm>0?std::sqrt(delta/norm):std::sqrt(delta)};
            require(receipts[layer].relative_error<1e-10,"connected units disagree with complete FFN");
            std::copy(output.begin(),output.end(),results+result_at);
            result_at+=d;
        }
        require(result_at==result_count,"result buffer does not match bound layer dimensions");
        return 0;
    } catch (const std::exception& e) {
        if (error && error_bytes) std::snprintf(error,error_bytes,"%s",e.what());
        return -1;
    }
}
