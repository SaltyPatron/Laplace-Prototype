/* Run a UCD *BreakTest.txt file through ICU's default UAX #29 iterator; report pass/fail per line. */
#include <unicode/ubrk.h>
#include <unicode/utext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv){
    UBreakIteratorType ty = !strcmp(argv[2], "grapheme") ? UBRK_CHARACTER : !strcmp(argv[2], "word") ? UBRK_WORD : UBRK_SENTENCE;
    FILE *f = fopen(argv[1], "r"); char line[8192]; int pass = 0, fail = 0, shown = 0;
    while (fgets(line, sizeof line, f)) {
        char *hash = strchr(line, '#'); if (hash) *hash = 0;
        char buf[4096]; int blen = 0, expect[512], ne = 0; char *p = line;
        while (*p) {
            if (!strncmp(p, "\xc3\xb7", 2)) { expect[ne++] = blen; p += 2; continue; }   /* ÷ break */
            if (!strncmp(p, "\xc3\x97", 2)) { p += 2; continue; }                          /* × no break */
            if ((*p >= '0' && *p <= '9') || (*p >= 'A' && *p <= 'F')) {
                unsigned cp = strtoul(p, &p, 16);
                if (cp < 0x80) buf[blen++] = cp;
                else if (cp < 0x800) { buf[blen++] = 0xC0 | cp >> 6; buf[blen++] = 0x80 | (cp & 0x3F); }
                else if (cp < 0x10000) { buf[blen++] = 0xE0 | cp >> 12; buf[blen++] = 0x80 | (cp >> 6 & 0x3F); buf[blen++] = 0x80 | (cp & 0x3F); }
                else { buf[blen++] = 0xF0 | cp >> 18; buf[blen++] = 0x80 | (cp >> 12 & 0x3F); buf[blen++] = 0x80 | (cp >> 6 & 0x3F); buf[blen++] = 0x80 | (cp & 0x3F); }
                continue;
            }
            p++;
        }
        if (!blen) continue;
        UErrorCode e = U_ZERO_ERROR; UText *ut = utext_openUTF8(NULL, buf, blen, &e);
        UBreakIterator *bi = ubrk_open(ty, "", NULL, 0, &e); ubrk_setUText(bi, ut, &e);
        int got[512], ng = 0; for (int32_t b = ubrk_first(bi); b != UBRK_DONE; b = ubrk_next(bi)) got[ng++] = b;
        int ok = ng == ne && !memcmp(got, expect, ne * sizeof(int));
        if (ok) pass++; else { fail++; if (shown++ < 3) fprintf(stderr, "  mismatch: %s\n", line); }
        ubrk_close(bi); utext_close(ut);
    }
    printf("%-22s %-9s pass %5d  fail %d\n", strrchr(argv[1], '/') + 1, argv[2], pass, fail);
    return fail != 0;
}
