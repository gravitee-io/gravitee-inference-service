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
package io.gravitee.inference.service.integration.embedding;

import static io.gravitee.inference.api.Constants.INFERENCE_FORMAT;
import static io.gravitee.inference.api.Constants.INFERENCE_TYPE;
import static io.gravitee.inference.api.Constants.SERVICE_INFERENCE_MODELS_ADDRESS;

import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.rest.openai.embedding.EncodingFormat;
import io.vertx.core.json.Json;
import java.io.IOException;
import java.net.URI;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.utility.DockerImageName;

public class ServiceOpenaiEmbeddingTest extends ServiceEmbeddingTest {

  private static final String MODEL_NAME = "all-minilm:latest";
  public static final String URI_K = "uri";
  public static final String API_KEY = "apiKey";
  public static final String MODEL = "model";
  public static final String ENCODING_FORMAT = "encodingFormat";

  static final String IMAGE_NAME = "ollama/ollama:0.1.26";
  public static final int PORT = 11434;
  static final OllamaContainer ollama = new OllamaContainer(DockerImageName.parse(IMAGE_NAME)).withExposedPorts(PORT);
  public static final String V_1 = "/v1";
  public static final String HTTP_PROTOCOL = "http://";

  @BeforeEach
  public void setup() throws IOException, InterruptedException {
    ollama.start();
    ollama.execInContainer("ollama", "pull", "all-minilm");
  }

  @AfterEach
  public void tearDown() {
    ollama.stop();
  }

  @Override
  String loadModel() {
    String host = "0.0.0.0";
    URI endpoint = URI.create(HTTP_PROTOCOL + host + ":" + PORT + V_1);

    InferenceRequest openaiStartRequest = new InferenceRequest(
      InferenceAction.START,
      Map.of(
        INFERENCE_FORMAT,
        "OPENAI",
        INFERENCE_TYPE,
        "EMBEDDING",
        URI_K,
        endpoint.toString(),
        API_KEY,
        "FAKE_KEY",
        MODEL,
        MODEL_NAME,
        ENCODING_FORMAT,
        EncodingFormat.FLOAT.name()
      )
    );

    return vertx
      .eventBus()
      .<Object>request(SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(openaiStartRequest))
      .blockingGet()
      .body()
      .toString();
  }
}
