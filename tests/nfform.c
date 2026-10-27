/* Report each file's Unicode normalization form (a source filter, never identity): NFC, NFD, both, or neither. */
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv){
    UErrorCode e = U_ZERO_ERROR;
    const UNormalizer2 *nfc = unorm2_getNFCInstance(&e), *nfd = unorm2_getNFDInstance(&e);
    for (int i = 1; i < argc; i++) {
        FILE *f = fopen(argv[i], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
        char *b = malloc(n); if (fread(b, 1, n, f) != (size_t)n) return 1; fclose(f);
        int32_t len = 0; e = U_ZERO_ERROR; u_strFromUTF8(NULL, 0, &len, b, n, &e); e = U_ZERO_ERROR;
        UChar *u = malloc(sizeof(UChar) * (len + 1)); u_strFromUTF8(u, len + 1, NULL, b, n, &e);
        UBool c = unorm2_isNormalized(nfc, u, len, &e), d = unorm2_isNormalized(nfd, u, len, &e);
        printf("%s\t%s\n", argv[i], c && d ? "both" : c ? "NFC" : d ? "NFD" : "neither");
        free(b); free(u);
    }
    return 0;
}
