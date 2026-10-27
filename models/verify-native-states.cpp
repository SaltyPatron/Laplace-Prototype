#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
template<class T> void read(std::ifstream&f,T*p,size_t n){if(!f.read(reinterpret_cast<char*>(p),n*sizeof(T)))throw std::runtime_error("truncated native state receipt");}
int main(int argc,char**argv)try{
 if(argc!=3&&argc!=4)throw std::runtime_error("usage: verify-native-states MODEL.gguf MODEL.states.bin [MODEL.targets.bin]");
 std::ifstream f(argv[2],std::ios::binary);char magic[8];int64_t rows,d;read(f,magic,8);read(f,&rows,1);read(f,&d,1);
 if(std::memcmp(magic,"LPSTATE1",8)||rows<4||d<1)throw std::runtime_error("invalid native state receipt");
 std::vector<float> expected(rows*d);read(f,expected.data(),expected.size());
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_mmap=true;
 auto*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("CPU model load failed");
 int n=llama_vocab_n_tokens(llama_model_get_vocab(model));if(rows<n+1||d!=llama_model_n_embd(model))throw std::runtime_error("native state dimensions differ");
 struct Target{double floor;std::map<int,double> delta;};std::map<int,Target> targets;
 if(argc==4){std::ifstream target(argv[3],std::ios::binary);char tag[8];int64_t outputs,count;read(target,tag,8);read(target,&outputs,1);read(target,&count,1);
  if(std::memcmp(tag,"LPTARG01",8)||outputs!=n||count<1||count>rows)throw std::runtime_error("invalid output target receipt");
  for(int64_t x=0;x<count;x++){int64_t node,length;double floor;read(target,&node,1);read(target,&floor,1);read(target,&length,1);
   if(node<1||node>=rows||length<1||length>n||!std::isfinite(floor)||targets.count((int)node))throw std::runtime_error("invalid candidate row");
   Target item{floor,{}};for(int64_t y=0;y<length;y++){int32_t id;double delta;read(target,&id,1);read(target,&delta,1);if(id<0||id>=n||!std::isfinite(delta)||!item.delta.emplace(id,delta).second)throw std::runtime_error("invalid candidate entry");}targets.emplace((int)node,std::move(item));
  }
 }
 std::vector<int> tokens(rows),parents(rows),positions(rows);int maxpos=1;
 std::ifstream index(std::string(argv[2])+".index.bin",std::ios::binary);
 if(index){char tag[8];int64_t count;read(index,tag,8);read(index,&count,1);if(std::memcmp(tag,"LPINDEX1",8)||count!=rows)throw std::runtime_error("invalid native execution index");
  for(int64_t x=0;x<rows;x++){read(index,&tokens[x],1);read(index,&parents[x],1);read(index,&positions[x],1);if(tokens[x]<0||tokens[x]>=n||parents[x]>=x)throw std::runtime_error("invalid prefix index");maxpos=std::max(maxpos,positions[x]);}
 }else{if(rows!=n+1)throw std::runtime_error("ordered state receipt requires execution index");tokens[0]=llama_vocab_bos(llama_model_get_vocab(model));parents[0]=-1;positions[0]=0;for(int x=1;x<=n;x++){tokens[x]=x-1;parents[x]=0;positions[x]=1;}}
 const int slots=64*(maxpos+1);
 constexpr int sequences=64;auto cp=llama_context_default_params();cp.n_ctx=slots;cp.n_seq_max=sequences;cp.n_batch=slots;cp.n_ubatch=slots;
 cp.n_threads=6;cp.n_threads_batch=6;cp.embeddings=true;cp.pooling_type=LLAMA_POOLING_TYPE_NONE;cp.attention_type=LLAMA_ATTENTION_TYPE_CAUSAL;
 cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto*ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("CPU context creation failed");auto batch=llama_batch_init(slots,0,1);
 double error=0,norm=0,maxabs=0,outside_sum=0,outside_max=0,logit_error=0,logit_norm=0;size_t values=0,measured_targets=0,candidate_winners=0,target_winners=0;
 std::vector<std::pair<double,int>> worst;
 for(int first=1;first<rows;first+=sequences){llama_memory_clear(llama_get_memory(ctx),true);int count=std::min((int)rows-first,sequences);batch.n_tokens=0;std::vector<int> outputs;
  for(int seq=0;seq<count;seq++){
   std::vector<int> history;for(int x=first+seq;x>=0;x=parents[x])history.push_back(x);std::reverse(history.begin(),history.end());
   for(int node:history){int i=batch.n_tokens++;batch.token[i]=tokens[node];batch.pos[i]=positions[node];batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=node==first+seq;}
   outputs.push_back(batch.n_tokens-1);
  }
  if(llama_decode(ctx,batch))throw std::runtime_error("CPU decode failed");
  for(int seq=0;seq<count;seq++){auto*a=llama_get_embeddings_ith(ctx,outputs[seq]);if(!a)throw std::runtime_error("CPU state missing");auto*b=expected.data()+(first+seq)*d;
   double rowerror=0;for(int c=0;c<d;c++){if(!std::isfinite(a[c])||!std::isfinite(b[c]))throw std::runtime_error("nonfinite state");double delta=a[c]-b[c];error+=delta*delta;rowerror+=delta*delta;norm+=(double)b[c]*b[c];maxabs=std::max(maxabs,std::abs(delta));++values;}worst.push_back({rowerror,first+seq});
   auto found=targets.find(first+seq);if(found!=targets.end()){
    const float* logits=llama_get_logits_ith(ctx,outputs[seq]);if(!logits)throw std::runtime_error("CPU logits missing");
    const Target&t=found->second;double maximum=-INFINITY,desired=-INFINITY,inside=0,outside=0;int winner=0;
    for(const auto&entry:t.delta)desired=std::max(desired,t.floor+entry.second);
    for(int y=0;y<n;y++){if(!std::isfinite(logits[y]))throw std::runtime_error("nonfinite CPU logit");maximum=std::max(maximum,double(logits[y]));if(logits[y]>logits[winner])winner=y;}
    for(int y=0;y<n;y++){auto candidate=t.delta.find(y);bool admitted=candidate!=t.delta.end();double target=t.floor+(admitted?candidate->second:0);double diff=logits[y]-target;logit_error+=diff*diff;logit_norm+=target*target;(admitted?inside:outside)+=std::exp(logits[y]-maximum);}
    double outside_mass=outside/(inside+outside);outside_sum+=outside_mass;outside_max=std::max(outside_max,outside_mass);++measured_targets;
    auto chosen=t.delta.find(winner);if(chosen!=t.delta.end()){++candidate_winners;if(std::abs(t.floor+chosen->second-desired)<1e-6)++target_winners;}
   }
  }
 }
 std::sort(worst.begin(),worst.end(),[](auto&a,auto&b){return a.first>b.first;});for(size_t x=0;x<std::min(size_t(10),worst.size());x++)std::cerr<<"state_error node="<<worst[x].second<<" token="<<tokens[worst[x].second]<<" squared="<<worst[x].first<<"\n";
 llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();double relative=std::sqrt(error/norm);
 std::cout<<"{\"vocabulary_states\":"<<n<<",\"execution_states\":"<<(rows-1)<<",\"values\":"<<values<<",\"relative_rms_error\":"<<relative<<",\"max_absolute_error\":"<<maxabs;
 if(measured_targets)std::cout<<",\"output_targets\":"<<measured_targets<<",\"candidate_top1_accuracy\":"<<double(candidate_winners)/measured_targets<<",\"target_top1_accuracy\":"<<double(target_winners)/measured_targets<<",\"mean_outside_mass\":"<<outside_sum/measured_targets<<",\"max_outside_mass\":"<<outside_max<<",\"relative_logit_error\":"<<std::sqrt(logit_error/logit_norm);
 std::cout<<",\"f32_kv\":true,\"state_comparison_passed\":"<<(relative<1e-4?"true":"false")<<"}\n";return relative<1e-4?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
