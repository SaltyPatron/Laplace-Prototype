// A calculation result, lowered through ordinary Unicode content and ordered
// composition. This is not an encoding of a checkpoint or a learned operator.
// All source arithmetic precedes this boundary. No candidate is thresholded.
#include <laplace/core/content_witness_batch.h>
#include <laplace/core/entity_type_law.h>
#include <laplace/core/mantissa.h>
#include <laplace/core/ordered_composition.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Component = laplace_ordered_component_t;
using Result = laplace_ordered_composition_result_t;
void check(int rc, const char* operation) {
    if (rc) throw std::runtime_error(operation);
}
Component component(const Result& r) {
    Component c{}; c.id=r.id; c.tier=r.tier;
    std::copy_n(r.coord,4,c.coord); return c;
}
Component text(intent_stage_t* stage, const char* value,
               const hash128_t& source, int64_t observed) {
    tier_tree_t* raw=nullptr;
    check(content_witness_tree_build(reinterpret_cast<const uint8_t*>(value),
                                    std::char_traits<char>::length(value),&raw),
          "canonical text decomposition failed");
    std::unique_ptr<tier_tree_t,decltype(&tier_tree_free)> tree(raw,tier_tree_free);
    tier_node_view_t node{};
    check(content_witness_tree_root_node(tree.get(),&node),"canonical text root missing");
    hash128_t root{};
    check(content_witness_emit_tree_at(stage,tree.get(),&source,nullptr,0,&root,observed),
          "canonical text staging failed");
    if (!hash128_equals(&root,&node.id)) throw std::runtime_error("text root changed during staging");
    Component c{};c.id=node.id;c.tier=node.tier;c.atom=node.atom;c.has_atom=node.tier==0;
    std::copy_n(node.coord,4,c.coord);return c;
}
Result compose(intent_stage_t* stage,const Component* children,size_t count,
               const hash128_t& type,const hash128_t& source,int64_t observed) {
    laplace_ordered_composition_request_t request{children,count,type,source,observed};
    Result result{};
    check(laplace_ordered_composition_stage_batch(stage,&request,1,&result),
          "ordered result composition failed");
    return result;
}
}

// Schema: [contract, ordered condition, relation, ordered scored responses].
// Each response is [canonical object, canonical decimal fp1e9 grade]. The
// response order is grade descending, then canonical object bytes. All of it
// is ordinary, reversible DAG content; the projection is a query carrier.
// Source, source-local circuit address and observation time never salt E.
// The caller owns/discards this stage on error; no partial result is publishable.
extern "C" int lp_model_conditional_record(
    intent_stage_t* stage,const char* contract,const char* relation,
    const Component* condition,size_t condition_count,
    const Component* objects,const int64_t* scores,size_t count,
    const hash128_t* source,int64_t observed,
    Result* out_condition,Result* out_response,Result* out_record,
    char* error,size_t error_bytes) {
    try {
        if(!stage||!contract||!*contract||!relation||!*relation||!condition||!condition_count||
           !objects||!scores||!count||count>INT32_MAX||!source||!out_condition||!out_response||!out_record)
            throw std::runtime_error("incomplete conditional calculation");
        hash128_t type{};check(laplace_entity_type_id("Source_Record",&type),"Source_Record is not governed");
        std::vector<size_t> order(count);
        for(size_t i=0;i<count;i++) {
            if(scores[i]<0||scores[i]>1000000000)throw std::runtime_error("grade outside fp1e9 unit interval");
            order[i]=i;
        }
        // Reject aliases rather than silently double-counting a candidate.
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return hash128_compare(&objects[a].id,&objects[b].id)<0;});
        for(size_t i=1;i<count;i++)if(hash128_equals(&objects[order[i-1]].id,&objects[order[i]].id))
            throw std::runtime_error("duplicate canonical response object");
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return scores[a]!=scores[b]?scores[a]>scores[b]:hash128_compare(&objects[a].id,&objects[b].id)<0;});

        auto recipe=text(stage,contract,*source,observed);
        auto predicate=text(stage,relation,*source,observed);
        std::map<int64_t,Component> scalar;
        std::vector<std::array<Component,2>> pairs(count);
        std::vector<laplace_ordered_composition_request_t> requests(count);
        std::vector<Result> results(count);
        std::vector<hash128_t> ids(count);std::vector<int64_t> grades(count);
        for(size_t i=0;i<count;i++) {
            size_t at=order[i];auto found=scalar.find(scores[at]);
            if(found==scalar.end()) {
                char digits[32];std::snprintf(digits,sizeof(digits),"%lld",static_cast<long long>(scores[at]));
                found=scalar.emplace(scores[at],text(stage,digits,*source,observed)).first;
            }
            pairs[i]={objects[at],found->second};
            requests[i]={pairs[i].data(),2,type,*source,observed};
            ids[i]=objects[at].id;grades[i]=scores[at];
        }
        check(laplace_ordered_composition_stage_batch(stage,requests.data(),count,results.data()),"scored response batch failed");
        std::vector<Component> responses(count);
        for(size_t i=0;i<count;i++)responses[i]=component(results[i]);
        Result input=compose(stage,condition,condition_count,type,*source,observed);
        Result response=compose(stage,responses.data(),count,type,*source,observed);
        Component fields[]={recipe,component(input),predicate,component(response)};
        // A singleton condition may itself be an atom; retain its declaration.
        if(condition_count==1)fields[1]=condition[0];
        Result record=compose(stage,fields,4,type,*source,observed);
        std::vector<double> walk(count*4);
        check(laplace_testimony_pack_walk(ids.data(),grades.data(),nullptr,count,walk.data()),"conditional testimony packing failed");
        hash128_t placement{};laplace_physicality_id_compute(record.id,3,&placement);
        check(intent_stage_add_physicality(stage,&placement,&record.id,3,record.coord,&record.hilbert,
              walk.data(),static_cast<uint32_t>(count),static_cast<int32_t>(count),1,0,0,1,observed),
              "conditional testimony staging failed");
        *out_condition=input;*out_response=response;*out_record=record;return 0;
    }catch(const std::exception& e){if(error&&error_bytes)std::snprintf(error,error_bytes,"%s",e.what());return -1;}
}
