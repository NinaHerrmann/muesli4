#! /bin/bash
rm -rf build/* \
## && mkdir build \
cd build \
&& cmake -DExample:STRING=examples/$1.cu -G "Unix Makefiles" .. \
&& make \
&& cd ..


