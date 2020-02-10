FROM openjdk:7
COPY . /usr/src/client
WORKDIR /usr/src/client/src/main/java
RUN javac fileClient.java
CMD ["java", "fileClient"]