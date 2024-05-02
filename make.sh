#!/bin/sh

set -xe
clang++ -Wall -Wextra -pedantic --std=gnu++2b pgrad.cc -o out/pgrad
rm -rf out/*.dSYM