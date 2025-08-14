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

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class BaseDownloadModelTest {

  protected static final Logger LOGGER = LoggerFactory.getLogger(BaseDownloadModelTest.class);
  private static final String HF_URL = "https://huggingface.co/";

  // Download file
  private static final String GRAVITEE_INFERENCE_SERVICE_DIRECTORY = "gravitee-inference-service";
  protected static final String TMP_DIR_NAME =
    System.getProperty("java.io.tmpdir") + "/" + GRAVITEE_INFERENCE_SERVICE_DIRECTORY;
  private static final Path TMP_DIR = Path.of(TMP_DIR_NAME);

  protected static URI getUriIfExist(String name, String modelPath) {
    try {
      if (Files.notExists(TMP_DIR)) {
        Files.createDirectory(TMP_DIR);
      }
      final Path fileDirectory = Path.of(TMP_DIR_NAME + "/" + name);
      if (Files.notExists(fileDirectory)) {
        Files.createDirectories(fileDirectory);
      }

      String[] modelPathSplit = modelPath.split("/");
      final Path file = Path.of(TMP_DIR_NAME + "/" + name + "/" + modelPathSplit[modelPathSplit.length - 1]);
      if (Files.notExists(file)) {
        LOGGER.info("Downloading [{}] from model [{}] in [{}]", modelPath, name, file.toAbsolutePath());
        Files.copy(URI.create(HF_URL + name + modelPath).toURL().openStream(), file, REPLACE_EXISTING);
        LOGGER.info("[{}] model downloaded", name);
      }
      return file.toUri();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
