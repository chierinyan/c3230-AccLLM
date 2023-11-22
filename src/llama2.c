/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME:
* NAME:
* UID :
* Development Platform:
* Remark: (How much you implemented?)
* How to compile: (gcc -o llama2_[UID] llama2_[UID].c utilities.c -O2 -pthread -lm)

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o llama2_[UID] llama2_[UID].c utilities.c -O2 -pthread -lm

Then Run with:
$ ./llama2_[UID] <seed> <thr_count>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 *
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is
 * independent of each row, so we can use Multi-Threading for acceleration.
 *
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 *
 * A sequential version is provided in seq.c, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>
#include <semaphore.h>

struct thr_arg {
    int id;
    int start;
    int end;
    int col;
    float* vec;
    float* mat;
    float* out;
    sem_t start_sem;
    sem_t end_sem;
};

// Global Variables
int _thr_count;
int finished = 0;
pthread_t tids[32];
struct thr_arg args[32];

double timeval_to_s(struct timeval* tv) {
    return tv->tv_sec + tv->tv_usec / 1000000.0;
}


void *thr_func(void *arg) {
    struct thr_arg* a = (struct thr_arg*) arg;
    while (1) {
        sem_wait(&a->start_sem);
        if (finished) break;

        for (int r = a->start; r < a->end; r++) {
            float row_out = 0.0f;
            for (int c = 0; c < a->col; c++) {
                row_out += a->vec[c] * a->mat[r*a->col+c];
            }
            a->out[r] = row_out;
        }
        sem_post(&a->end_sem);
    }

    struct rusage usage;
    getrusage(RUSAGE_THREAD, &usage);
    printf("thread %d has completed - user: %f s, system: %f s\n", a->id,
           timeval_to_s(&usage.ru_utime), timeval_to_s(&usage.ru_stime));
    return NULL;
}

int init_mat_vec_mul(int thr_count) {
    _thr_count = thr_count;
    for (int i = 0; i < thr_count; i++) {
        args[i].id = i;
        sem_init(&args[i].start_sem, 0, 0);
        sem_init(&args[i].end_sem, 0, 0);
        pthread_create(&tids[i], NULL, thr_func, &args[i]);
    }
    return 0;
}

void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    int row_per_thr = (row+_thr_count-1)/_thr_count; // floor(row, thr_count)
    for (int thr = 0; thr < _thr_count; thr+=1) {
        args[thr].start = thr*row_per_thr;
        args[thr].end = (thr+1)*row_per_thr<row ? (thr+1)*row_per_thr : row;
        args[thr].col = col;
        args[thr].vec = vec;
        args[thr].mat = mat;
        args[thr].out = out;
        sem_post(&args[thr].start_sem);
    }
    for (int thr=0; thr<_thr_count; thr++) {
        sem_wait(&args[thr].end_sem);
    }
}

int close_mat_vec_mul() {
    finished = 1;
    for (int thr = 0; thr < _thr_count; thr++) {
        sem_post(&args[thr].start_sem);
        pthread_join(tids[thr], NULL);
        sem_destroy(&args[thr].start_sem);
    }

    struct rusage main_usage;
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %f s, system: %f s\n",
           timeval_to_s(&main_usage.ru_utime), timeval_to_s(&main_usage.ru_stime));
    return 0;
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {

    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);

            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }

        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }

    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);

    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}
