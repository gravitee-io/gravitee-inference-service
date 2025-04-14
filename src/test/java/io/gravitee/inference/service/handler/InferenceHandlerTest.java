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

import static io.gravitee.inference.api.service.InferenceAction.INFER;
import static io.gravitee.inference.api.service.InferenceAction.START;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import io.gravitee.inference.api.Constants;
import io.gravitee.inference.api.InferenceModel;
import io.gravitee.inference.api.classifier.ClassifierResult;
import io.gravitee.inference.api.classifier.ClassifierResults;
import io.gravitee.inference.api.embedding.EmbeddingTokenCount;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.repository.Model;
import io.reactivex.rxjava3.core.Observable;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.EventBus;
import io.vertx.rxjava3.core.eventbus.Message;
import io.vertx.rxjava3.core.eventbus.MessageConsumer;
import java.util.Comparator;
import java.util.Map;
import java.util.TreeSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
@ExtendWith(MockitoExtension.class)
public class InferenceHandlerTest {

  @Mock
  private Vertx vertx;

  @Mock
  private EventBus eventBus;

  @Mock
  private MessageConsumer<Buffer> messageConsumer;

  @Mock
  private InferenceModel<Object, String, Object> model;

  @Mock
  private Message<Buffer> message;

  private InferenceHandler inferenceHandler;

  @BeforeEach
  public void setUp() {
    when(vertx.eventBus()).thenReturn(eventBus);
    when(eventBus.<Buffer>consumer(anyString())).thenReturn(messageConsumer);
    when(messageConsumer.toObservable()).thenReturn(Observable.just(message));

    inferenceHandler = new InferenceHandler("test-address", new Model(0, model), vertx);
  }

  @Test
  public void must_handle_inference_request_with_classifier() {
    String input = "The big brown fox jumps over the lazy dog";

    var collection = new TreeSet<>(Comparator.comparingDouble(ClassifierResult::score));
    collection.add(new ClassifierResult("Negative", 0.20f));
    collection.add(new ClassifierResult("Positive", 0.90f));

    InferenceRequest request = new InferenceRequest(INFER, Map.of(Constants.INPUT, input));
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    when(model.infer(input)).thenReturn(new ClassifierResults(collection));

    inferenceHandler.handle(message);

    ArgumentCaptor<Buffer> captor = ArgumentCaptor.forClass(Buffer.class);
    verify(message).reply(captor.capture());
    Buffer response = captor.getValue();

    var actual = Json.decodeValue(response, ClassifierResults.class);
    assertNotNull(actual);
  }

  @Test
  public void must_handle_inference_request_with_embedding() {
    String input = "The big brown fox jumps over the lazy dog";

    var embedding = new EmbeddingTokenCount(new float[] { 0, 1, 2, 3, 4, 5 }, 53);

    InferenceRequest request = new InferenceRequest(INFER, Map.of(Constants.INPUT, input));
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    when(model.infer(input)).thenReturn(embedding);

    inferenceHandler.handle(message);

    ArgumentCaptor<Buffer> captor = ArgumentCaptor.forClass(Buffer.class);
    verify(message).reply(captor.capture());
    Buffer response = captor.getValue();

    var actual = Json.decodeValue(response, EmbeddingTokenCount.class);
    assertEquals(embedding.tokenCount(), actual.tokenCount());
    assertArrayEquals(embedding.embedding(), actual.embedding());
  }

  @Test
  public void must_handle_unsupported_inference_request() {
    InferenceRequest request = new InferenceRequest(START, Map.of(Constants.INPUT, "some-input"));
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    inferenceHandler.handle(message);

    verify(message).fail(405, "Unsupported action: START");
  }

  @Test
  public void must_handle_exception_inference_request() {
    when(message.body()).thenThrow(new RuntimeException("Test exception"));

    inferenceHandler.handle(message);

    verify(message).fail(400, "Test exception");
  }

  @AfterEach
  public void afterEach() {
    inferenceHandler.close();
  }
}
