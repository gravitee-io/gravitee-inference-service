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
package io.gravitee.inference.service.handler;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.api.utils.ConfigWrapper;
import io.gravitee.inference.onnx.OnnxInference;
import io.gravitee.inference.service.model.LocalModelFactory;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.eventbus.Message;
import java.util.Map;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class LocalInferenceHandler implements InferenceHandler {

  private final int key;
  private final LocalModelFactory localModelFactory;

  private OnnxInference<?, ?, ?> model;
  private final Map<String, Object> payload;

  public LocalInferenceHandler(Map<String, Object> payload, LocalModelFactory modelFactory) {
    this.payload = payload;
    this.localModelFactory = modelFactory;
    this.key = payload.hashCode();
  }

  @Override
  public void handle(Message<Buffer> message) {
    try {
      var request = Json.decodeValue(message.body(), InferenceRequest.class);
      switch (request.action()) {
        case INFER -> {
          var config = new ConfigWrapper(request.payload());
          var output = model.infer(config.get(Constants.INPUT));
          message.reply(Json.encodeToBuffer(output));
        }
        case null, default -> message.fail(405, "Unsupported action: " + request.action());
      }
    } catch (Exception e) {
      message.fail(400, e.getMessage());
    }
  }

  public void close() {
    if (model != null) {
      model.close();
    }
  }

  @Override
  public int key() {
    return key;
  }

  @Override
  public void loadModel() {
    model = localModelFactory.build(new ConfigWrapper(payload));
  }
}
