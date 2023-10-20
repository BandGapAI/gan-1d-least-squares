#!/bin/bash

for f in *.pdf; do
  echo transforming $f
  pdftops -eps -r 600 $f $(basename -s .pdf $f).eps
done
