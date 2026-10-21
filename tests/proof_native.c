/* Independent reader of persisted entity/physicality rows. The production
 * ingester is a separate process; this verifier uses only libpq and Native. */
#include <laplace/laplace.h>
#include <laplace/geometry.h>
#include <laplace/highway.h>
#include <laplace/tier0.h>
#include <libpq-fe.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { lp_ref value; uint8_t *path; size_t bytes; } Stored;
static void require(int ok,const char *what){ if (!ok) { fprintf(stderr,"FAIL %s\n",what); exit(1); } }
static uint8_t *unhex(const char *hex,size_t *size){
    *size=strlen(hex)/2; uint8_t *out=malloc(*size+1); require(out!=NULL,"hex allocation");
    for (size_t i=0; i<*size; i++) { unsigned value; require(sscanf(hex+2*i,"%2x",&value)==1,"hex byte"); out[i]=(uint8_t)value; }
    return out;
}
static bool coordinate(void *context,const lp_ewkb_span *span){
    double xyzm[4]; return span->count==1 && lp_ewkb_vertex(span,0,xyzm) && lp_coord_of_xyzm(xyzm,context);
}
static lp_ref child(lp_idmap *nodes,const lp_tier0_record *tier0,const lp_id *parent,const lp_id *id,int *missing){
    int64_t at=lp_idmap_find(nodes,id);
    if (at>=0) return ((Stored *)lp_idmap_at(nodes,(size_t)at))->value;
    int64_t cp=lp_tier0_codepoint(tier0,id);
    if (cp<0) { char p[33],c[33]; lp_hex(parent->b,16,p); lp_hex(id->b,16,c);
        fprintf(stderr,"missing child %s of persisted parent %s\n",c,p); *missing=1; return (lp_ref){0}; }
    return lp_ref_atom(tier0,(uint32_t)cp);
}
static uint32_t component(uint32_t *parent,uint32_t at){
    while (parent[at]!=at) { parent[at]=parent[parent[at]]; at=parent[at]; } return at;
}
static void database(const char *conninfo,const lp_tier0_record *tier0){
    PGconn *pg=PQconnectdb(conninfo); require(PQstatus(pg)==CONNECTION_OK,PQerrorMessage(pg));
    PGresult *result=PQexec(pg,"SELECT e.id::text,e.tier,encode(ST_AsEWKB(e.coord),'hex'),encode(ST_AsEWKB(p.path),'hex') FROM public.entity e JOIN public.physicality p ON p.entity=e.id");
    require(PQresultStatus(result)==PGRES_TUPLES_OK,PQerrorMessage(pg));
    int rows=PQntuples(result); require(rows>0,"real persisted compositions");
    lp_idmap *nodes=lp_idmap_sized(sizeof(Stored));
    for (int i=0; i<rows; i++) {
        lp_id id; require(lp_id_unhex(PQgetvalue(result,i,0),&id),"stored BLAKE3 ID");
        bool fresh; size_t at=lp_idmap_put(nodes,&id,&fresh); require(fresh,"one physicality per entity in proof database");
        Stored *node=lp_idmap_at(nodes,at); node->value.id=id; node->value.tier=(uint8_t)atoi(PQgetvalue(result,i,1));
        size_t size; uint8_t *point=unhex(PQgetvalue(result,i,2),&size);
        require(lp_ewkb_visit(point,size,coordinate,&node->value.c,NULL),"persisted POINT ZM coordinate"); free(point);
        node->path=unhex(PQgetvalue(result,i,3),&node->bytes);
    }
    PQclear(result); size_t wrong_height=0,invalid=0;
    for (size_t i=0; i<lp_idmap_count(nodes); i++) {
        Stored *node=lp_idmap_at(nodes,i);
        lp_path path=lp_path_of(node->path,node->bytes); size_t count=lp_path_len(path);
        require(count>=2,"every composed path has at least two logical children");
        lp_ref *parts=calloc(count,sizeof *parts); require(parts!=NULL,"path allocation");
        size_t at=0; uint8_t height=0; int missing=0;
        for (size_t v=0; v<path.n; v++) {
            lp_id id=lp_path_id(path,v); lp_ref value=child(nodes,tier0,&node->value.id,&id,&missing);
            if (value.tier>=height) height=(uint8_t)(value.tier+1);
            for (uint32_t run=lp_path_run(path,v); run; run--) parts[at++]=value;
        }
        require(at==count,"ordered run expansion");
        if (missing) { char id[33]; lp_hex(node->value.id.b,16,id); fprintf(stderr,"invalid parent %s: missing child\n",id);
            free(parts); invalid++; continue; }
        lp_ref expected=lp_ref_compose(parts,count,height); free(parts);
        require(lp_id_eq(&expected.id,&node->value.id),"persisted ID hashes its actual ordered children");
        require(!memcmp(expected.c.m,node->value.c.m,sizeof expected.c.m),"persisted coordinate composes from actual child coordinates");
        if (height!=node->value.tier) { char id[33]; lp_hex(node->value.id.b,16,id);
            fprintf(stderr,"invalid parent %s: tier %u expected %u\n",id,node->value.tier,height); wrong_height++; }
    }
    printf("database: %zu persisted IDs and coordinates verified; %zu tier-height violations\n",(size_t)rows-invalid,wrong_height);
    for (size_t i=0; i<lp_idmap_count(nodes); i++) free(((Stored *)lp_idmap_at(nodes,i))->path);
    lp_idmap_free(nodes); PQfinish(pg); require(!invalid,"every child is recorded or a scalar leaf"); require(!wrong_height,"tier=max(child.tier)+1");
}
static void cache(const lp_tier0_record *tier0,const char *path){
    const lp_highway *h=lp_highway_map(path); require(h!=NULL,"single-file Highway mapping");
    lp_text *text=lp_text_new(tier0); require(text!=NULL,"Native UAX decomposer");
    size_t count=0,edges=0;
    uint32_t *parent=malloc(h->nrec*sizeof *parent); uint16_t *namespaces=calloc(h->nrec,sizeof *namespaces);
    require(parent && namespaces,"interchange graph workspace");
    for (size_t i=0; i<h->nrec; i++) parent[i]=(uint32_t)i;
    const char *names[]={"ili","Synset","fnframe","fnfe","fnlu","vnclass","pbroleset","vaframe","language"};
    for (size_t i=0; i<sizeof names/sizeof *names; i++) require(lp_highway_list(h,names[i])!=NULL,"all required Highway namespaces");
    for (size_t li=0; li<h->nlists; li++) for (uint32_t slot=0; slot<h->list[li].count; slot++) {
        const lp_list *list=h->list+li; const lp_tier0_record *record=lp_highway_at(h,list,slot);
        if (!record) continue;
        const char *id=lp_highway_identifier(h,list,slot); require(id!=NULL,"embedded identifier text");
        const char *const *parts; size_t np=lp_highway_identifier_parts(h,list,slot,&parts);
        require(np>0,"embedded identifier components"); lp_ref *p=calloc(np,sizeof *p); require(p!=NULL,"identifier components allocation");
        for (size_t j=0; j<np; j++) p[j]=lp_text_decompose(text,(const uint8_t *)parts[j],strlen(parts[j]),NULL,NULL);
        lp_ref value=np==1 ? p[0] : lp_ref_compose(p,np,lp_tier_above(p,np)); free(p);
        require(lp_id_eq(&value.id,&record->id) && !memcmp(value.c.m,record->m,sizeof record->m),"canonical Highway identifier");
        require((np>1 || lp_highway_key(h,list,id)==slot) && lp_highway_slot(h,list,&value.id)==slot,"constant-time slot and ID lookup"); count++;
        for (size_t to=0; to<h->nlists; to++) { const lp_edge *out; size_t n=lp_highway_edges(h,list->name,slot,h->list[to].name,&out); edges+=n;
            for (size_t e=0; e<n; e++) {
                require(lp_highway_at(h,h->list+to,out[e].to)!=NULL,"crosswalk target exists");
                uint32_t a=component(parent,list->first+slot),b=component(parent,h->list[to].first+out[e].to);
                if (a!=b) parent[b]=a;
            }
        }
    }
    lp_mask packed=0;
    for (size_t i=0; i<h->nbanks; i++) require(lp_mask_put(&packed,h->bank+i,h->bank[i].list->count-1),"pack disjoint 64-bit banks");
    for (size_t i=0; i<h->nbanks; i++) require(lp_mask_has(packed,h->bank+i,h->bank[i].list->count-1),"filter every packed bank");
    require(edges==h->nedges,"all CSR edges are reachable");
    for (size_t ns=0; ns<8; ns++) {
        const lp_list *list=lp_highway_list(h,names[ns]);
        for (uint32_t i=0; i<list->count; i++) if (lp_highway_at(h,list,i))
            namespaces[component(parent,list->first+i)]|=(uint16_t)(1u<<ns);
    }
    bool connected=false;
    for (size_t i=0; i<h->nrec; i++) if (namespaces[i]==255) { connected=true; break; }
    require(connected,"real crosswalks connect all eight identifier namespaces");
    printf("highway interchange: all eight identifier namespaces connected by real crosswalks\n");
    const lp_list *language=lp_highway_list(h,"language");
    require(lp_highway_key(h,language,"en")==lp_highway_key(h,language,"eng"),"ISO 639 two/three-letter interchange");
    free(parent); free(namespaces);
    uint8_t hash[32]; char hex[65]; lp_highway_fingerprint(h,hash); lp_hex(hash,32,hex);
    printf("highway: %zu identifiers, %zu edges, %zu 64-bit banks; BLAKE3 %s\n",count,edges,h->nbanks,hex);
    lp_text_free(text);
}
int main(int argc,char **argv){
    require(argc==4,"usage: proof_native cache|database TIER0 HIGHWAY|CONNINFO");
    const lp_tier0_record *tier0=lp_tier0_map(argv[2]); require(tier0!=NULL,"canonical scalar Tier-0 cache");
    FILE *f=fopen(argv[2],"rb"); require(f!=NULL,"Tier-0 file"); require(!fseek(f,0,SEEK_END),"Tier-0 size");
    require(ftell(f)==(long)((size_t)LP_TIER0_COUNT*sizeof *tier0) && LP_TIER0_COUNT==1112064,"1,112,064 scalar records"); fclose(f);
    if (!strcmp(argv[1],"cache")) cache(tier0,argv[3]);
    else if (!strcmp(argv[1],"database")) database(argv[3],tier0);
    else require(0,"unknown proof command");
    return 0;
}
