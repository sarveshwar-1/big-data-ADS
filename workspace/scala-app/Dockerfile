FROM eclipse-temurin:11-jdk-jammy

# Install sbt
ARG SBT_VERSION=1.9.7
RUN apt-get update && \
    apt-get install -y curl && \
    curl -L -o sbt-$SBT_VERSION.deb https://repo.scala-sbt.org/scalasbt/debian/sbt-$SBT_VERSION.deb && \
    dpkg -i sbt-$SBT_VERSION.deb && \
    rm sbt-$SBT_VERSION.deb && \
    apt-get clean

WORKDIR /app
CMD ["sbt", "clean", "assembly"]
