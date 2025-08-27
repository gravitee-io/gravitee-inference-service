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
package io.gravitee.inference.service.repository;

import io.gravitee.inference.service.handler.InferenceHandler;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class HandlerRepository implements Repository<InferenceHandler> {

  private static final Logger LOGGER = LoggerFactory.getLogger(HandlerRepository.class);

  private final Map<Integer, InferenceHandler> models = new ConcurrentHashMap<>();
  private final Map<Integer, AtomicInteger> counters = new ConcurrentHashMap<>();

  @Override
  public InferenceHandler add(InferenceHandler handler) {
    models.compute(
      handler.key(),
      (k, v) -> {
        if (v != null) {
          LOGGER.debug("Model already exists, returning existing model");
          counters.computeIfPresent(
            k,
            (__, cv) -> {
              cv.incrementAndGet();
              return cv;
            }
          );
          return v;
        }

        LOGGER.debug("Model does not exist, creating model");

        counters.put(k, new AtomicInteger(1));

        handler.loadModel();

        return handler;
      }
    );

    return models.get(handler.key());
  }

  public int getModelsSize() {
    return models.size();
  }

  public int getModelUsage(int key) {
    return counters.containsKey(key) ? counters.get(key).get() : 0;
  }

  @Override
  public void remove(InferenceHandler handler) {
    counters.computeIfPresent(
      handler.key(),
      (k, v) -> {
        var counter = v.decrementAndGet();
        if (counter == 0) {
          LOGGER.debug("Model not in use anymore, tearing down model");
          handler.close();
          models.remove(k);
          LOGGER.debug("Model successfully removed");
          return null;
        }
        LOGGER.debug("Model still in use [{} time(s)]", counter);
        return v;
      }
    );
  }
}
