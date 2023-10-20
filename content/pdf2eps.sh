#!/bin/bash

files=( cost gh_test_case_c2_1_g0_9_h1_5.pdf )

for f in "${files[@]}"; do
  echo transforming $f
  pdftops -eps -r 600 $f.pdf $f.eps
done
