# Builder
FROM nvcr.io/nvidia/pytorch:23.07-py3 AS builder

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add rustup to PATH
ENV PATH=/root/.cargo/bin:$PATH
RUN rustup install stable

# Build
COPY ./ /prog
WORKDIR /prog
RUN cargo build --release

# Build actual
FROM nvcr.io/nvidia/pytorch:23.07-py3

# Build correct compute arches
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX;8.9;9.0"

# Compile AutoGPTQ
RUN git clone https://github.com/PanQiWei/AutoGPTQ.git
WORKDIR AutoGPTQ
RUN pip install '.[triton]'
WORKDIR /
RUN rm -rf AutoGPTQ

# Compile CTranslate2
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git
WORKDIR CTranslate2
RUN mkdir build
WORKDIR build
RUN cmake -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF ..
RUN make -j16
RUN make install
WORKDIR ../python
RUN pip install -r install_requirements.txt
RUN python setup.py bdist_wheel
RUN pip install dist/*.whl
WORKDIR /
RUN rm -rf CTranslate2
RUN pip install hf-hub-ctranslate2>=2.12.0

# Install server
COPY --from=builder /prog/target/release/llmcode-server /
RUN chmod +x /llmcode-server

ENTRYPOINT ["/llmcode-server"]