#include "llama.h"
#include "gguf.h"
#include "laplace/synthesis/lookup_compile.h"
#include "laplace/synthesis/init.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void read(std::ifstream &stream,T *out,size_t count) {
    if(!stream.read(reinterpret_cast<char*>(out),count*sizeof(T))) throw std::runtime_error("truncated readout program");
}

int main(int argc,char **argv) try {
    if(argc!=3&&argc!=4) throw std::runtime_error("usage: laplace-readout-calibrate MODEL.gguf PROGRAM.bin [--evaluate]");
    const bool evaluate=argc==4&&std::string(argv[3])=="--evaluate";
    if(argc==4&&!evaluate) throw std::runtime_error("unknown evaluation option");
    const auto begin=std::chrono::steady_clock::now();
    if(laplace_synthesis_init()!=0) throw std::runtime_error("native synthesis initialization failed");
    std::ifstream source(argv[2],std::ios::binary);
    char magic[8];read(source,magic,8);
    if(std::memcmp(magic,"LPREAD01",8)) throw std::runtime_error("unknown readout program");
    int32_t n,d,nnz;read(source,&n,1);read(source,&d,1);read(source,&nnz,1);
    if(n<=0||d<=0||nnz<0) throw std::runtime_error("invalid program dimensions");
    std::vector<int> input(n),emit(n),rows(nnz),cols(nnz);
    std::vector<double> weights(nnz);
    read(source,input.data(),n);read(source,emit.data(),n);read(source,rows.data(),nnz);
    read(source,cols.data(),nnz);read(source,weights.data(),nnz);
    if(source.peek()!=EOF) throw std::runtime_error("unexpected data after readout program");
    auto *container=gguf_init_from_file(argv[1],{true,nullptr});
    if(!container) throw std::runtime_error("cannot read model container");
    int64_t tensor=gguf_find_tensor(container,"output.weight");
    if(tensor<0||gguf_get_tensor_type(container,tensor)!=GGML_TYPE_F32||
       gguf_get_tensor_size(container,tensor)!=(size_t)n*d*sizeof(float))
        throw std::runtime_error("output.weight must be a matching F32 matrix");
    size_t offset=gguf_get_data_offset(container)+gguf_get_tensor_offset(container,tensor);
    gguf_free(container);
    llama_backend_init();
    auto params=llama_model_default_params();
    params.n_gpu_layers=0;params.use_mmap=true;
    auto *model=llama_model_load_from_file(argv[1],params);
    if(!model) throw std::runtime_error("CPU model load failed");
    if(llama_model_n_embd(model)!=d||llama_vocab_n_tokens(llama_model_get_vocab(model))!=n)
        throw std::runtime_error("model/program dimensions differ");
    constexpr int sequences=128;
    auto context_params=llama_context_default_params();
    context_params.n_ctx=sequences*8;context_params.n_seq_max=sequences;
    context_params.n_batch=sequences*2;context_params.n_ubatch=sequences*2;
    context_params.n_threads=6;context_params.n_threads_batch=6;
    context_params.embeddings=!evaluate;context_params.pooling_type=LLAMA_POOLING_TYPE_NONE;
    context_params.attention_type=LLAMA_ATTENTION_TYPE_CAUSAL;
    auto *context=llama_init_from_model(model,context_params);
    if(!context) throw std::runtime_error("CPU context creation failed");
    auto batch=llama_batch_init(sequences*2,0,1);
    std::vector<float> states((size_t)n*d),head((size_t)n*d);
    std::vector<double> targets;
    std::vector<bool> active(n,false);
    size_t correct=0,active_count=0,active_correct=0;
    if(evaluate) {
        std::vector<double> identity((size_t)n*n);targets.resize((size_t)n*n);
        int rc=laplace_lookup_compile(n,n,2,0,input.data(),emit.data(),rows.data(),cols.data(),weights.data(),nnz,identity.data(),targets.data());
        if(rc) throw std::runtime_error("invalid evaluation target: "+std::to_string(rc));
        for(int e=0;e<nnz;e++) if(weights[e]>0&&input[rows[e]]!=input[cols[e]]&&emit[input[cols[e]]]>=0) active[input[rows[e]]]=true;
    }
    const llama_token bos=llama_vocab_bos(llama_model_get_vocab(model));
    if(bos<0) throw std::runtime_error("model has no BOS token");
    for(int first=0;first<n;first+=sequences) {
        llama_memory_clear(llama_get_memory(context),true);
        int count=std::min(sequences,n-first);batch.n_tokens=count*2;
        for(int seq=0;seq<count;seq++) for(int pos=0;pos<2;pos++) {
            int i=seq*2+pos;batch.token[i]=pos?first+seq:bos;
            batch.pos[i]=pos;batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=pos==1;
        }
        int rc=llama_decode(context,batch);
        if(rc) throw std::runtime_error("CPU decode failed: "+std::to_string(rc));
        for(int seq=0;seq<count;seq++) {
            if(evaluate) {
                const float *logits=llama_get_logits_ith(context,seq*2+1);
                if(!logits) throw std::runtime_error("runtime omitted selected output logits");
                size_t best=0;double desired=-INFINITY;int operand=input[first+seq];
                for(int y=0;y<n;y++) {
                    if(!std::isfinite(logits[y]))throw std::runtime_error("nonfinite runtime logit");
                    if(logits[y]>logits[best])best=y;
                    desired=std::max(desired,targets[(size_t)y*n+operand]);
                }
                bool match=std::abs(targets[best*n+operand]-desired)<1e-8;
                correct+=match;
                if(active[operand]){active_count++;active_correct+=match;}
                continue;
            }
            const float *state=llama_get_embeddings_ith(context,seq*2+1);
            if(!state) throw std::runtime_error("runtime omitted a selected hidden state");
            for(int c=0;c<d;c++) if(!std::isfinite(state[c])) throw std::runtime_error("nonfinite runtime hidden state");
            std::copy(state,state+d,states.begin()+(size_t)(first+seq)*d);
        }
        std::cerr<<"measured "<<first+count<<"/"<<n<<" complete-model hidden states\n";
    }
    llama_batch_free(batch);llama_free(context);llama_model_free(model);llama_backend_free();
    if(evaluate) {
        std::cout<<"{\"method\":\"complete CPU transformer logits; no refit\",\"vocabulary_states\":"<<n
            <<",\"top1_target_accuracy\":"<<(double)correct/n<<",\"active_states\":"<<active_count
            <<",\"active_top1_target_accuracy\":"<<(active_count?(double)active_correct/active_count:0)
            <<",\"empty_states\":"<<n-active_count<<",\"empty_top1_target_accuracy\":"<<(n>active_count?(double)(correct-active_correct)/(n-active_count):0)<<"}\n";
        return 0;
    }
    size_t rank=0;double residual=0,accuracy=0;
    int rc=laplace_readout_fit(states.data(),n,d,input.data(),emit.data(),rows.data(),cols.data(),weights.data(),nnz,
        head.data(),&rank,&residual,&accuracy);
    if(rc) throw std::runtime_error("native readout solve failed: "+std::to_string(rc));
    for(float value:head) if(!std::isfinite(value)) throw std::runtime_error("nonfinite solved output weight");
    std::fstream target(argv[1],std::ios::binary|std::ios::in|std::ios::out);
    target.seekp(offset);target.write(reinterpret_cast<const char*>(head.data()),head.size()*sizeof(float));
    target.flush();if(!target) throw std::runtime_error("output weight write failed");
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<"{\"method\":\"SGELSD on complete CPU transformer states\",\"vocabulary_states\":"<<n
        <<",\"hidden_width\":"<<d<<",\"numerical_rank\":"<<rank<<",\"relative_logit_error\":"<<residual
        <<",\"top1_target_accuracy\":"<<accuracy<<",\"output_weight_offset\":"<<offset
        <<",\"output_weight_bytes\":"<<head.size()*sizeof(float)<<",\"seconds\":"<<seconds<<"}\n";
    return 0;
} catch(const std::exception &error) {
    std::cerr<<error.what()<<"\n";return 1;
}
