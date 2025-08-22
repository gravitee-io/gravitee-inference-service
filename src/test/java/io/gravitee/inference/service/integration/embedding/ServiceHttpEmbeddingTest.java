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

@Disabled
public class ServiceHttpEmbeddingTest extends ServiceEmbeddingTest {

  @Override
  String loadModel() {
    InferenceRequest httpStartRequest = new InferenceRequest(
      InferenceAction.START,
      Map.of(
        INFERENCE_FORMAT,
        "HTTP",
        INFERENCE_TYPE,
        "EMBEDDING",
        "uri",
        "http://localhost:8000/embed",
        "method",
        "POST",
        "headers",
        Map.of("Content-Type", "application/json"),
        "requestBodyTemplate",
        "{\"text\": \"\"}",
        "inputLocation",
        "$.text",
        "outputEmbeddingLocation",
        "$.embedding"
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
