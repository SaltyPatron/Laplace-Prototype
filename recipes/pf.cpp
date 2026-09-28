// preflate CLI: "decode in.deflate out.unpacked out.diff" and "encode in.unpacked in.diff out.deflate" (bit-exact).
#include <cstdio>
#include <cstring>
#include <vector>
#include "preflate.h"
static std::vector<unsigned char> rd(const char *p){ std::vector<unsigned char> d; FILE*f=fopen(p,"rb"); int c; while((c=fgetc(f))!=EOF) d.push_back(c); fclose(f); return d; }
static void wr(const char *p, const std::vector<unsigned char>&d){ FILE*f=fopen(p,"wb"); fwrite(d.data(),1,d.size(),f); fclose(f); }
int main(int argc, char **argv){
  if (argc == 5 && !strcmp(argv[1], "decode")) { std::vector<unsigned char> un, diff; auto raw = rd(argv[2]);
    if (!preflate_decode(un, diff, raw)) { fprintf(stderr, "decode failed\n"); return 1; } wr(argv[3], un); wr(argv[4], diff); return 0; }
  if (argc == 5 && !strcmp(argv[1], "encode")) { std::vector<unsigned char> re; auto un = rd(argv[2]); auto diff = rd(argv[3]);
    if (!preflate_reencode(re, diff, un)) { fprintf(stderr, "encode failed\n"); return 1; } wr(argv[4], re); return 0; }
  fprintf(stderr, "usage: pf decode|encode ...\n"); return 2;
}
