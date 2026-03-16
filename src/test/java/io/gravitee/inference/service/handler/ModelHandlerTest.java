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

import static io.gravitee.inference.api.Constants.INFERENCE_FORMAT;
import static io.gravitee.inference.api.Constants.MODEL_ADDRESS_KEY;
import static io.gravitee.inference.api.service.InferenceAction.*;
import static io.reactivex.rxjava3.core.Observable.fromRunnable;
import static io.reactivex.rxjava3.core.Observable.timer;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

import io.gravitee.inference.api.memory.InsufficientVramException;
import io.gravitee.inference.api.memory.MemoryEstimate;
import io.gravitee.inference.api.service.InferenceFormat;
import io.gravitee.inference.api.service.InferenceRequest;
import io.gravitee.inference.service.provider.InferenceHandlerProvider;
import io.gravitee.inference.service.provider.ModelProviderRegistry;
import io.gravitee.inference.service.repository.HandlerRepository;
import io.gravitee.reactive.webclient.api.ModelFetcher;
import io.gravitee.reactive.webclient.api.ModelFileType;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.Single;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.json.Json;
import io.vertx.rxjava3.core.Vertx;
import io.vertx.rxjava3.core.eventbus.EventBus;
import io.vertx.rxjava3.core.eventbus.Message;
import io.vertx.rxjava3.core.eventbus.MessageConsumer;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
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
public class ModelHandlerTest {

  @Mock
  private EventBus eventBus;

  @Mock
  private Vertx vertx;

  @Mock
  private ModelFetcher fetcher;

  @Mock
  private HandlerRepository repository;

  @Mock
  private ModelProviderRegistry modelProviderRegistry;

  @Mock
  private InferenceHandlerProvider modelProvider;

  @Mock
  private MessageConsumer<Buffer> messageConsumer;

  @Mock
  private Message<Buffer> message;

  private ModelHandler modelHandler;

  private io.vertx.core.Vertx delegate;

  @BeforeEach
  public void setUp() {
    delegate = io.vertx.core.Vertx.vertx();
    lenient().when(vertx.getDelegate()).thenReturn(delegate);
    lenient().when(vertx.eventBus()).thenReturn(eventBus);

    lenient()
      .when(eventBus.<Buffer>consumer(anyString()))
      .thenReturn(messageConsumer);
    Observable<Message<Buffer>> observable = Observable.just(message);

    lenient().when(messageConsumer.toObservable()).thenReturn(observable);
    lenient().when(message.body()).thenReturn(Buffer.buffer("some_address"));
    lenient()
      .when(fetcher.fetchModel(any()))
      .thenReturn(
        Single.just(
          Map.of(
            ModelFileType.CONFIG,
            "/path/to/config.json",
            ModelFileType.TOKENIZER,
            "/path/to/tokenizer.json",
            ModelFileType.MODEL,
            "/path/to/model.json"
          )
        )
      );

    lenient()
      .when(modelProviderRegistry.getProvider(any(InferenceFormat.class)))
      .thenReturn(modelProvider);
    lenient()
      .when(modelProvider.provide(any(), any()))
      .thenReturn(Single.just(mock(InferenceHandler.class)));

    modelHandler = new ModelHandler(vertx, repository, modelProviderRegistry);
  }

  @Test
  public void must_handle_create_model_action() {
    HashMap<String, Object> payload = new HashMap<>();
    payload.put("modelName", "models/");
    payload.put(INFERENCE_FORMAT, InferenceFormat.ONNX_BERT);

    InferenceRequest request = new InferenceRequest(START, payload);
    when(message.body()).thenReturn(Json.encodeToBuffer(request));
    InferenceHandler model = mock(InferenceHandler.class);
    when(modelProvider.provide(any(), any())).thenReturn(Single.just(model));

    fromRunnable(() -> modelHandler.handle(message))
      .flatMap(__ -> timer(2, TimeUnit.SECONDS))
      .test()
      .awaitDone(3, TimeUnit.SECONDS)
      .assertComplete()
      .assertNoErrors();

    ArgumentCaptor<Buffer> captor = ArgumentCaptor.forClass(Buffer.class);
    verify(message, times(1)).reply(captor.capture());
    Buffer addressBuffer = captor.getValue();

    doNothing().when(repository).remove(any());
    when(message.body()).thenReturn(
      Json.encodeToBuffer(
        new InferenceRequest(
          STOP,
          Map.of(MODEL_ADDRESS_KEY, addressBuffer.toString())
        )
      )
    );

    fromRunnable(() -> modelHandler.handle(message))
      .flatMap(__ -> timer(2, TimeUnit.SECONDS))
      .test()
      .awaitDone(3, TimeUnit.SECONDS)
      .assertComplete()
      .assertNoErrors();

    verify(message, times(2)).reply(any());
  }

  @Test
  public void must_handle_stop_model_action() {
    InferenceRequest request = new InferenceRequest(
      STOP,
      Map.of(MODEL_ADDRESS_KEY, "unknownAddress")
    );
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    modelHandler.handle(message);

    verify(message).fail(
      400,
      "Could not find inference handler for address: unknownAddress"
    );
  }

  @Test
  public void must_handle_unsupported_format() {
    InferenceRequest request = new InferenceRequest(INFER, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    modelHandler.handle(message);

    verify(message).fail(405, "Unsupported action: INFER");
  }

  @Test
  public void must_handle_null() {
    InferenceRequest request = new InferenceRequest(null, new HashMap<>());
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    modelHandler.handle(message);

    verify(message).fail(405, "Unsupported action: null");
  }

  @Test
  public void must_handle_unknown_exception() {
    when(message.body()).thenThrow(new RuntimeException("Test exception"));

    modelHandler.handle(message);

    verify(message).fail(400, "Test exception");
  }

  @Test
  public void must_handle_insufficient_vram_with_503() {
    HashMap<String, Object> payload = new HashMap<>();
    payload.put("modelName", "models/");
    payload.put(INFERENCE_FORMAT, InferenceFormat.VLLM);

    InferenceRequest request = new InferenceRequest(START, payload);
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    // Simulate InsufficientVramException from provider (Qwen3-0.6B on a 4 GiB GPU)
    MemoryEstimate estimate = new MemoryEstimate(
      4.0,
      3.6,
      1.5,
      false,
      "Model may not fit. Try gpu_memory_utilization=0.42, or reduce max_model_len.",
      true
    );
    InsufficientVramException vramEx = new InsufficientVramException(
      "Qwen/Qwen3-0.6B",
      estimate
    );
    when(modelProvider.provide(any(), any())).thenReturn(Single.error(vramEx));

    fromRunnable(() -> modelHandler.handle(message))
      .flatMap(__ -> timer(2, TimeUnit.SECONDS))
      .test()
      .awaitDone(3, TimeUnit.SECONDS)
      .assertComplete()
      .assertNoErrors();

    ArgumentCaptor<Integer> codeCaptor = ArgumentCaptor.forClass(Integer.class);
    ArgumentCaptor<String> msgCaptor = ArgumentCaptor.forClass(String.class);
    verify(message, atLeast(1)).fail(codeCaptor.capture(), msgCaptor.capture());

    // The ModelHandler error callback is the one that sets 503
    assertThat(codeCaptor.getAllValues()).contains(503);
    String vramMsg = msgCaptor
      .getAllValues()
      .stream()
      .filter(s -> s.contains("Insufficient VRAM"))
      .findFirst()
      .orElseThrow();
    assertThat(vramMsg).contains("does NOT fit");
    assertThat(vramMsg).contains("1.50");
    assertThat(vramMsg).contains("4.00");
  }

  @Test
  public void must_handle_generic_provider_error_with_500() {
    HashMap<String, Object> payload = new HashMap<>();
    payload.put("modelName", "models/");
    payload.put(INFERENCE_FORMAT, InferenceFormat.VLLM);

    InferenceRequest request = new InferenceRequest(START, payload);
    when(message.body()).thenReturn(Json.encodeToBuffer(request));

    when(modelProvider.provide(any(), any())).thenReturn(
      Single.error(new RuntimeException("Python crash"))
    );

    fromRunnable(() -> modelHandler.handle(message))
      .flatMap(__ -> timer(2, TimeUnit.SECONDS))
      .test()
      .awaitDone(3, TimeUnit.SECONDS)
      .assertComplete()
      .assertNoErrors();

    ArgumentCaptor<Integer> codeCaptor2 = ArgumentCaptor.forClass(
      Integer.class
    );
    ArgumentCaptor<String> msgCaptor2 = ArgumentCaptor.forClass(String.class);
    verify(message, atLeast(1)).fail(
      codeCaptor2.capture(),
      msgCaptor2.capture()
    );

    assertThat(codeCaptor2.getAllValues()).contains(500);
    assertThat(
      msgCaptor2
        .getAllValues()
        .stream()
        .filter(s -> s.contains("Python crash"))
        .findFirst()
    ).isPresent();
  }

  @AfterEach
  public void tearDown() {
    modelHandler.close();
  }
}
