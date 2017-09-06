#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy as np
cimport numpy as np
from cpython cimport array
import array


# cdef extern from "voidptr.h":
#     void* PyCObject_AsVoidPtr(object obj)

ctypedef np.float32_t REAL_t
DEF MAX_SENTENCE_LEN = 10000
cdef REAL_t ONEF = <REAL_t>1.0

cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

def data_iterator_cython(model, sentences, alpha):
    cdef int d
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef array.array indices = array.array('i', [0,0,0,0,0,0,0])
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)


    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index

            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    cdef np.uint32_t target_index = 0
    cdef unsigned long long modulo = 281474976710655ULL
    # release GIL & train on all sentences
    # with nogil:
    for sent_idx in range(effective_sentences):
        idx_start = sentence_idx[sent_idx]
        idx_end = sentence_idx[sent_idx + 1]
        for i in range(idx_start, idx_end):
            j = i - window + reduced_windows[i]
            if j < idx_start:
                j = idx_start
            k = i + window + 1 - reduced_windows[i]
            if k > idx_end:
                k = idx_end
            for j in range(j, k):
                if j == i:
                    continue
                indices[0] = indexes[j]
                indices[1] = indexes[i]
                for d in range(negative):
                    target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                    next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
                    while (indices[1] == target_index):
                        target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
                        next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
                    indices[d+2] = target_index

                yield indices