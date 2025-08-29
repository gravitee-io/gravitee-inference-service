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
package io.gravitee.inference.service.integration.classification.token;

import static io.gravitee.inference.api.Constants.*;
import static io.gravitee.inference.service.provider.HuggingFaceProvider.MODEL_NAME;

import io.gravitee.inference.api.classifier.ClassifierMode;
import io.gravitee.inference.api.service.InferenceAction;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.service.InferenceType;
import io.gravitee.inference.service.integration.classification.sequence.ServiceSequenceClassificationTest;
import io.vertx.core.json.Json;
import java.util.List;
import java.util.Map;

class BertTokenClassificationTest extends ServiceTokenClassificationTest {

  @Override
  String loadModel() {
    InferenceRequest localStartRequest = new InferenceRequest(
      InferenceAction.START,
      Map.ofEntries(
        Map.entry(MODEL_NAME, "dslim/distilbert-NER"),
        Map.entry(INFERENCE_FORMAT, InferenceFormat.ONNX_BERT),
        Map.entry(INFERENCE_TYPE, InferenceType.CLASSIFIER),
        Map.entry(CLASSIFIER_MODE, ClassifierMode.TOKEN),
        Map.entry(MODEL_PATH, "onnx/model.onnx"),
        Map.entry(TOKENIZER_PATH, "tokenizer.json"),
        Map.entry(CONFIG_JSON_PATH, "config.json"),
        Map.entry(DISCARDED_LABELS, List.of("O"))
      )
    );

    return vertx
      .eventBus()
      .<Object>request(SERVICE_INFERENCE_MODELS_ADDRESS, Json.encodeToBuffer(localStartRequest))
      .blockingGet()
      .body()
      .toString();
  }

  @Override
  Integer waitTime() {
    return 25_000;
  }
}
