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
import org.springframework.lang.Nullable;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class HandlerRepository implements Repository<InferenceHandler> {

  private static final Logger LOGGER = LoggerFactory.getLogger(
    HandlerRepository.class
  );

  private final Map<Integer, ModelEntry> models = new ConcurrentHashMap<>();

  @Override
  public InferenceHandler add(InferenceHandler handler) {
    return models
      .compute(handler.key(), (k, v) -> {
        if (v == null) {
          LOGGER.debug("Model does not exist, creating model");
          ModelEntry entry = new ModelEntry(handler);
          entry.handler().loadModel();
          return entry;
        }
        LOGGER.debug("Model already exists, returning existing model");
        return v.retain();
      })
      .handler();
  }

  public int getModelsSize() {
    return models.size();
  }

  public int getModelUsage(int key) {
    return models
      .getOrDefault(key, new ModelEntry(null, new AtomicInteger()))
      .counter()
      .get();
  }

  @Override
  public void remove(InferenceHandler handler) {
    models.computeIfPresent(handler.key(), (k, v) -> v.release());
  }

  private record ModelEntry(InferenceHandler handler, AtomicInteger counter) {
    private ModelEntry(InferenceHandler handler) {
      this(handler, new AtomicInteger(1));
    }

    public ModelEntry retain() {
      counter.incrementAndGet();
      return this;
    }

    @Nullable
    public ModelEntry release() {
      int i = counter.decrementAndGet();
      if (i <= 0) {
        LOGGER.debug("Model not in use anymore, tearing down model");
        handler.close();
        return null;
      }
      LOGGER.debug("Model still in use [{} time(s)]", counter);
      return this;
    }
  }
}
