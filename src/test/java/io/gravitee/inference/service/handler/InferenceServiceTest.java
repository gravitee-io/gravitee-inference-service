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

import static org.junit.jupiter.api.Assertions.*;

import io.gravitee.inference.service.InferenceService;
import io.vertx.rxjava3.core.Vertx;
import org.junit.jupiter.api.Test;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class InferenceServiceTest {

  @Test
  public void must_not_load_inference_service() throws Exception {
    Vertx vertx = Vertx.vertx();
    var inferenceService = new InferenceService(vertx, false);
    inferenceService.doStart();

    assertFalse(inferenceService.enabled());
    assertNull(inferenceService.crudHandler());

    inferenceService.doStop();
    vertx.close();
  }

  @Test
  public void must_load_inference_service() throws Exception {
    Vertx vertx = Vertx.vertx();
    var inferenceService = new InferenceService(vertx, true);
    inferenceService.doStart();

    assertTrue(inferenceService.enabled());
    assertNotNull(inferenceService.crudHandler());

    inferenceService.doStop();
    vertx.close();
  }
}
