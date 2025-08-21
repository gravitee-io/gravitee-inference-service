/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.inference.service;

import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class ServiceEmbeddingTest {

  protected static final Logger LOGGER = LoggerFactory.getLogger(ServiceEmbeddingTest.class);

  public static final String INPUT = "input";

  protected Vertx vertx;

  @BeforeEach
  public void setUp() throws Exception {
    vertx = Vertx.vertx();
    String modelPath = Files.createDirectories(Path.of("models")).toFile().getAbsolutePath();
    InferenceService inferenceService = new InferenceService(vertx, modelPath);
    inferenceService.start();
    Thread.sleep(2000);
  }

  @AfterEach
  public void tearDown() {
    vertx.close();
  }

  abstract String loadModel();

  Integer waitTime() {
    return 0;
  }

  static Stream<String> testInputProvider() {
    return Stream.of(
      "Hello world",
      "Hello, this is a test sentence for embedding generation.",
      "This is a much longer text to test how the embedding service handles longer inputs with multiple sentences and various punctuation marks. It includes different types of content and should produce a meaningful embedding vector.",
      "AI"
    );
  }

  @ParameterizedTest
  @MethodSource("testInputProvider")
  void shouldPerformInference(String inputText) throws InterruptedException {
    String modelAddress = loadModel();

    System.out.println("Model started at address: " + modelAddress);

    Thread.sleep(waitTime());

    InferenceRequest inferRequest = new InferenceRequest(InferenceAction.INFER, Map.of(INPUT, inputText));

    var embeddingTokenCountTestObserver = vertx
      .eventBus()
      .<Object>request(modelAddress, Json.encodeToBuffer(inferRequest))
      .map(objectMessage -> objectMessage.body().toString())
      .map(object -> Json.decodeValue(object, EmbeddingTokenCount.class))
      .doOnError(error -> LOGGER.error("Error during embedding inference for input '{}': {}", inputText, error.getMessage()))
      .test();

    embeddingTokenCountTestObserver
      .awaitDone(Duration.ofSeconds(30).toMillis(), java.util.concurrent.TimeUnit.MILLISECONDS)
      .assertComplete()
      .assertNoErrors()
      .assertValue(embeddingTokenCount -> embeddingTokenCount.embedding().length > 0)
      .assertValue(embeddingTokenCount -> embeddingTokenCount.tokenCount() > 0);
  }
}
