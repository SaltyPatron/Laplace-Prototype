// Connected FFN decomposition over held SafeTensors source handles.
// The managed caller supplies the production recognizer's bindings. No tokenizer,
// tensor-name convention, persistent operator records or vocabulary pair scan.
#include <laplace/dynamics/model_math.h>
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
