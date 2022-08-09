#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpdecimal.h>

_Decimal64 __attribute__((noinline, noclone))
dfp_add(_Decimal64 a, _Decimal64 b) {
    return a + b;
}


_Decimal64 __attribute__((noinline, noclone))
dfp_sub(_Decimal64 a, _Decimal64 b) {
    return a - b;
}

_Decimal64 __attribute__((noinline, noclone))
dfp_div(_Decimal64 a, _Decimal64 b) {
    return a / b;
}

_Decimal64 __attribute__((noinline, noclone))
dfp_mult(_Decimal64 a, _Decimal64 b) {
    return a * b;
}


int main(int argc, char **argv) {
/************************************* Add, SET 1 ************************************************/
    mpd_context_t ctx;
    mpd_t *a, *b;
    mpd_t *result;
    char *rstring;
    char status_str[MPD_MAX_FLAG_STRING];
    clock_t start_clock, end_clock;
    mpd_init(&ctx, 38);
    ctx.traps = 0;
    result = mpd_new(&ctx);
    a = mpd_new(&ctx);
    b = mpd_new(&ctx);

//98765.4	4.3201

    mpd_set_string(a, "98765.4", &ctx);
    mpd_set_string(b, "4.3201", &ctx);
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        mpd_add(result, a, b, &ctx);
    end_clock = clock();
    printf("Test Cases\nValue 1: 1.23456\nValue 2: 5.6789\n");
    printf("Test Case 1: Addition\n");
    fprintf(stderr, "Libmpdec time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);
    rstring = mpd_to_sci(result, 1);
    mpd_snprint_flags(status_str, MPD_MAX_FLAG_STRING, ctx.status);
    printf("\nResult: %s  %s\n", rstring, status_str);

    _Decimal64 aa = 98765.4DD;
    _Decimal64 bb = 4.3201DD;
    _Decimal64 rresult = aa;
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        rresult = dfp_add(rresult, bb);
    end_clock = clock();
    fprintf(stderr, "Optimized time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);

    printf("\nResult: %lx\n", (long) rresult);



    printf("Test Case 3: Subtraction\n");
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        mpd_sub(result, a, b, &ctx);
    end_clock = clock();
    fprintf(stderr, "Libmpdec time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);
    rstring = mpd_to_sci(result, 1);
    mpd_snprint_flags(status_str, MPD_MAX_FLAG_STRING, ctx.status);
    printf("\nResult: %s  %s\n", rstring, status_str);
    rresult = aa;
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        rresult = dfp_sub(rresult, bb);
    end_clock = clock();
    fprintf(stderr, "Optimized time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);

    printf("\nResult: %lx\n", (long) rresult);


    printf("Test Case 3: Multiplication\n");
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        mpd_mul(result, a, b, &ctx);
    end_clock = clock();
    fprintf(stderr, "Libmpdec time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);
    rstring = mpd_to_sci(result, 1);
    mpd_snprint_flags(status_str, MPD_MAX_FLAG_STRING, ctx.status);
    printf("\nResult: %s  %s\n", rstring, status_str);
    rresult = aa;
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        rresult = dfp_mult(rresult, bb);
    end_clock = clock();
    fprintf(stderr, "Optimized time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);

    printf("\nResult: %lx\n", (long) rresult);

    mpd_set_string(a, "98765.4", &ctx);
    mpd_set_string(b, "4.3201", &ctx);
    aa = 98765.4DD;
    bb  = 4.3201DD;




    printf("Test Case 4: Division\n");
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        mpd_div(result, a, b, &ctx);
    end_clock = clock();
    fprintf(stderr, "Libmpdec time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);
    rstring = mpd_to_sci(result, 1);
    mpd_snprint_flags(status_str, MPD_MAX_FLAG_STRING, ctx.status);
    printf("\nResult: %s  %s\n", rstring, status_str);
    rresult = aa;
    start_clock = clock();
    for (int i = 0; i < 1000000; i++)
        rresult = dfp_div(aa, bb);
    end_clock = clock();
    fprintf(stderr, "Optimized time: %f\n",
            (double) (end_clock - start_clock) / (double) CLOCKS_PER_SEC);

    printf("\nResult: %lx\n", (long) rresult);
    mpd_del(a);
    mpd_del(b);
    mpd_del(result);
    mpd_free(rstring);
}
