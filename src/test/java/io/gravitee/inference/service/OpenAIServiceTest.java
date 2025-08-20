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

import static io.gravitee.inference.api.Constants.*;
import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.rest.openai.embedding.EncodingFormat;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import java.io.File;
import java.nio.file.Files;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class OpenAIServiceTest {

  private static final String SERVICE_URL = "http://localhost:11434/v1";
  private static final String MODEL_NAME = "all-minilm:latest";
  public static final String INPUT = "input";
  public static final String URI = "uri";
  public static final String API_KEY = "apiKey";
  public static final String MODEL = "model";
  public static final String ENCODING_FORMAT = "encodingFormat";

  private Vertx vertx;

  @BeforeEach
  public void setUp() throws Exception {
    vertx = Vertx.vertx();
    File modelPath = Files.createTempDirectory("inference-test-models").toFile();
    InferenceService inferenceService = new InferenceService(vertx, modelPath.getAbsolutePath());
    inferenceService.start();
    Thread.sleep(2000);
  }

  @AfterEach
  public void tearDown() {
    vertx.close();
  }

  @Test
  void shouldPerformInferenceWhenOllamaIsAvailable() throws InterruptedException {
    CountDownLatch inferLatch = new CountDownLatch(1);
    AtomicReference<Object> inferResult = new AtomicReference<>();
    AtomicReference<Throwable> inferError = new AtomicReference<>();

    InferenceRequest startRequest = new InferenceRequest(
      InferenceAction.START,
      Map.of(
        INFERENCE_FORMAT,
        "OPENAI",
        INFERENCE_TYPE,
        "EMBEDDING",
        URI,
        java.net.URI.create(SERVICE_URL),
        API_KEY,
        "FAKE_KEY",
        MODEL,
        MODEL_NAME,
        ENCODING_FORMAT,
        EncodingFormat.FLOAT.name()
      )
    );

    String modelAddress = vertx
      .eventBus()
      .<Object>request(SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(startRequest))
      .blockingGet()
      .body()
      .toString();

    System.out.println("Model started at address: " + modelAddress);

    InferenceRequest inferRequest = new InferenceRequest(
      InferenceAction.INFER,
      Map.of(INPUT, "Hello, this is a test sentence for embedding generation.")
    );

    var embeddingTokenCountTestObserver = vertx
      .eventBus()
      .<Object>request(modelAddress, Json.encodeToBuffer(inferRequest))
      .map(objectMessage -> objectMessage.body().toString())
      .map(object -> Json.decodeValue(object, EmbeddingTokenCount.class))
      .test();

    embeddingTokenCountTestObserver
      .await()
      .assertComplete()
      .assertValue(embedding -> {
        assertThat(embedding).isNotNull();
        assertThat(embedding.embedding()).isNotEmpty();

        for (float value : embedding.embedding()) {
          assertThat(Float.isFinite(value)).isTrue();
        }

        return true;
      });
  }
}
