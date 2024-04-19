FROM ubuntu:20.04

ARG target_profile=release

# Install requirements
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install curl gcc g++ -y

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
