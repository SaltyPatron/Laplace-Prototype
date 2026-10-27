#include "gguf.h"
#include "laplace/synthesis/gguf_writer.h"
#include "laplace/synthesis/safetensors_parser.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using Json=nlohmann::json;
void check(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv)try{
 check(argc==4,"usage: safetensors-consumer SAFETENSORS_DIR TOKENIZER_REFERENCE.gguf VERIFICATION.gguf");
 std::string dir=argv[1];Json cfg,receipts;std::ifstream(dir+"/config.json")>>cfg;std::ifstream(dir+".tensors.json")>>receipts;
 auto*reference=gguf_init_from_file(argv[2],{true,nullptr});check(reference,"reference GGUF metadata unavailable");
 for(const auto&field:std::map<std::string,std::string>{{"hidden_size","embedding_length"},{"num_hidden_layers","block_count"},{"intermediate_size","feed_forward_length"},{"num_attention_heads","attention.head_count"},{"num_key_value_heads","attention.head_count_kv"},{"vocab_size","vocab_size"}}){
  auto key="llama."+field.second;auto i=gguf_find_key(reference,key.c_str());check(i>=0&&cfg.at(field.first).get<uint32_t>()==gguf_get_val_u32(reference,i),"verification reference differs at "+field.first);
 }
 auto*writer=gguf_writer_create(argv[3]);check(writer,"cannot create verification container");
 for(int64_t i=0;i<gguf_get_n_kv(reference);i++){
  const char*name=gguf_get_key(reference,i);std::string key=name;if(!(key.starts_with("tokenizer.")||key.starts_with("llama.")||key=="general.architecture"))continue;int rc=-1;
  switch(gguf_get_kv_type(reference,i)){
   case GGUF_TYPE_STRING:rc=gguf_writer_add_metadata_str(writer,name,gguf_get_val_str(reference,i));break;
   case GGUF_TYPE_UINT32:rc=gguf_writer_add_metadata_u32(writer,name,gguf_get_val_u32(reference,i));break;
   case GGUF_TYPE_FLOAT32:rc=gguf_writer_add_metadata_f32(writer,name,gguf_get_val_f32(reference,i));break;
   case GGUF_TYPE_BOOL:rc=gguf_writer_add_metadata_bool(writer,name,gguf_get_val_bool(reference,i));break;
   case GGUF_TYPE_ARRAY:{auto count=gguf_get_arr_n(reference,i);auto type=gguf_get_arr_type(reference,i);
    if(type==GGUF_TYPE_STRING){std::vector<uint8_t> packed;for(size_t x=0;x<count;x++){const char*s=gguf_get_arr_str(reference,i,x);uint64_t len=std::strlen(s);for(int b=0;b<8;b++)packed.push_back((uint8_t)(len>>(8*b)));packed.insert(packed.end(),s,s+len);}rc=gguf_writer_add_metadata_str_array_packed(writer,name,packed.data(),packed.size(),count);}
    else if(type==GGUF_TYPE_FLOAT32)rc=gguf_writer_add_metadata_f32_array(writer,name,static_cast<const float*>(gguf_get_arr_data(reference,i)),count);
    else if(type==GGUF_TYPE_INT32)rc=gguf_writer_add_metadata_i32_array(writer,name,static_cast<const int32_t*>(gguf_get_arr_data(reference,i)),count);
   }break;
   default:break;
  }
  check(rc==0,"unsupported reference metadata "+key);
 }
 gguf_free(reference);check(gguf_writer_add_metadata_str(writer,"general.name","SafeTensors CPU verification container")==0,"verification metadata failure");
 std::ifstream shard(dir+"/model.safetensors",std::ios::binary);uint64_t length=0;shard.read(reinterpret_cast<char*>(&length),8);check(length<64*1024*1024,"invalid SafeTensors header length");
 std::vector<char> bytes(8+length);std::memcpy(bytes.data(),&length,8);shard.read(bytes.data()+8,length);check((bool)shard,"truncated SafeTensors header");
 auto*header=safetensors_parse_header(bytes.data(),bytes.size());check(header,"native SafeTensors parser refused source");std::map<std::string,int> source;
 for(int i=0;i<safetensors_tensor_count(header);i++)source[safetensors_tensor_name(header,i)]=i;
 size_t read_count=0,write_count=0;uint64_t parameters=0;
 auto read_tensor=[&](const std::string&name,const std::vector<size_t>&shape){auto it=source.find(name);check(it!=source.end(),"missing HF tensor "+name);int i=it->second;check(std::string(safetensors_tensor_dtype(header,i))=="F32","verification requires F32");check(safetensors_tensor_rank(header,i)==(int)shape.size(),"HF rank mismatch "+name);size_t count=1;
  for(size_t a=0;a<shape.size();a++){check(safetensors_tensor_dim(header,i,a)==(int64_t)shape[a],"HF axis mismatch "+name);count*=shape[a];}
  check(safetensors_tensor_data_end(header,i)-safetensors_tensor_data_start(header,i)==(int64_t)(count*4),"HF byte count mismatch "+name);
  std::vector<float> values(count);shard.seekg(safetensors_header_bytes(header)+safetensors_tensor_data_start(header,i));shard.read(reinterpret_cast<char*>(values.data()),count*4);check((bool)shard,"truncated source tensor "+name);for(float x:values)check(std::isfinite(x),"nonfinite source tensor "+name);++read_count;return values;
 };
 for(const auto&receipt:receipts){if(receipt["gguf_name"].is_null())continue;auto name=receipt.at("gguf_name").get<std::string>();auto shape=receipt.at("shape").get<std::vector<size_t>>();std::vector<float> values;
  if(!receipt["hf_name"].is_null())values=read_tensor(receipt.at("hf_name").get<std::string>(),shape);
  else{
   check(shape.size()==3,"unknown packed GGUF tensor "+name);auto end=name.find('.',4);auto layer=name.substr(4,end-4);std::string slot=name.find("ffn_gate_exps")!=std::string::npos?"w1":name.find("ffn_up_exps")!=std::string::npos?"w3":name.find("ffn_down_exps")!=std::string::npos?"w2":"";check(!slot.empty(),"unknown expert packing "+name);
   for(size_t e=0;e<shape[0];e++){auto part=read_tensor("model.layers."+layer+".block_sparse_moe.experts."+std::to_string(e)+"."+slot+".weight",{shape[1],shape[2]});values.insert(values.end(),part.begin(),part.end());}
  }
  parameters+=values.size();std::reverse(shape.begin(),shape.end());check(gguf_writer_add_tensor(writer,name.c_str(),0,shape.data(),shape.size(),values.data())==0,"verification writer refused "+name);++write_count;
 }
 check(read_count==source.size(),"source tensors were omitted from verification");safetensors_header_free(header);check(gguf_writer_finalize(writer)==0,"verification container finalization failed");gguf_writer_free(writer);
 std::cout<<"{\"source_tensors\":"<<read_count<<",\"consumer_tensors\":"<<write_count<<",\"parameters\":"<<parameters<<",\"all_source_tensors_consumed\":true}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
