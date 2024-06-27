FROM ubuntu:20.04

ARG target_profile=release

# Install requirements (gcc must be at least version 11)
RUN apt-get update && apt-get upgrade -y && apt-get install software-properties-common -y && apt-get clean
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install curl gcc-11 g++-11 -y && apt-get clean
RUN ln -s /usr/bin/gcc-11 /usr/bin/gcc
RUN ln -s /usr/bin/gcc-11 /usr/bin/cc
RUN ln -s /usr/bin/g++-11 /usr/bin/g++

# Install Rust for root
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y
ENV PATH="/root/.cargo/bin:$PATH"

# Copy the source code
COPY . /app

# Set the working directory for the compilation
WORKDIR /app/bin

# Compile the applications
RUN ./build_all.sh ${target_profile}

# Set the entrypoint
ENTRYPOINT ["./run_tests.sh"]
