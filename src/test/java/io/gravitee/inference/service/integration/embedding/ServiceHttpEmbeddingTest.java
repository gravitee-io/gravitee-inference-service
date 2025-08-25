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

import static io.gravitee.inference.api.Constants.*;

import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceRequest;
import io.vertx.core.json.Json;
import java.util.Map;
import org.junit.jupiter.api.Disabled;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.utility.DockerImageName;

@Disabled
public class ServiceHttpEmbeddingTest extends ServiceEmbeddingTest {

  public static final String URI = "uri";
  public static final String METHOD = "method";
  public static final String HEADERS = "headers";
  public static final String REQUEST_BODY_TEMPLATE = "requestBodyTemplate";
  public static final String INPUT_LOCATION = "inputLocation";
  public static final String OUTPUT_EMBEDDING_LOCATION = "outputEmbeddingLocation";

  private static final String MODEL_NAME = "all-minilm:latest";

  static final String IMAGE_NAME = "ollama/ollama:0.1.26";
  public static final int PORT = 11434;
  static final OllamaContainer ollama = new OllamaContainer(DockerImageName.parse(IMAGE_NAME)).withExposedPorts(PORT);

  @Override
  String loadModel() {
    String serviceUrl = String.format("http://%s:%d/v1/embeddings", ollama.getHost(), PORT);
    InferenceRequest httpStartRequest = new InferenceRequest(
      InferenceAction.START,
      Map.of(
        INFERENCE_FORMAT,
        "HTTP",
        INFERENCE_TYPE,
        "EMBEDDING",
        URI,
        serviceUrl,
        METHOD,
        "POST",
        HEADERS,
        Map.of("Content-Type", "application/json", "Authorization", "Bearer: FAKEAPIKEY"),
        REQUEST_BODY_TEMPLATE,
        String.format("{\"input\": \"\", \"model\":\"%s\"}", MODEL_NAME),
        INPUT_LOCATION,
        "$.input",
        OUTPUT_EMBEDDING_LOCATION,
        "$.data[-1].embedding"
      )
    );

    return vertx
      .eventBus()
      .<Object>request(SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(httpStartRequest))
      .blockingGet()
      .body()
      .toString();
  }

  @Override
  Integer waitTime() {
    return 5_000;
  }
}
