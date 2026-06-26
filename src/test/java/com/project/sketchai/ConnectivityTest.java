package com.project.sketchai;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import com.project.sketchai.service.AiService;

import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

@SpringBootTest
public class ConnectivityTest {

    @Value("${python.ai.engine.url:http://localhost:8000}")
    private String pythonEngineUrl;

    @Autowired
    private AiService aiService;

    @Test
    public void testConnectivity() {
        System.out.println("=== Starting Connectivity Audit from Spring Boot ===");
        
        int connectTimeout = -1;
        int readTimeout = -1;
        int maxAttempts = 3; // default from code
        
        try {
            // Use reflection to extract RestTemplate from AiService
            Field rtField = AiService.class.getDeclaredField("restTemplate");
            rtField.setAccessible(true);
            RestTemplate restTemplate = (RestTemplate) rtField.get(aiService);
            
            if (restTemplate.getRequestFactory() instanceof SimpleClientHttpRequestFactory) {
                SimpleClientHttpRequestFactory factory = (SimpleClientHttpRequestFactory) restTemplate.getRequestFactory();
                
                Field ctField = SimpleClientHttpRequestFactory.class.getDeclaredField("connectTimeout");
                ctField.setAccessible(true);
                connectTimeout = (int) ctField.get(factory);
                
                Field rtFieldFactory = SimpleClientHttpRequestFactory.class.getDeclaredField("readTimeout");
                rtFieldFactory.setAccessible(true);
                readTimeout = (int) rtFieldFactory.get(factory);
            }
        } catch (Exception e) {
            System.err.println("Could not extract timeout configurations via reflection: " + e.getMessage());
        }

        System.out.println("Configured Python Backend URL: " + pythonEngineUrl);
        System.out.println("Connect Timeout (ms): " + connectTimeout);
        System.out.println("Read Timeout (ms): " + readTimeout);
        System.out.println("Retry Max Attempts: " + maxAttempts);

        // Test actual connection
        RestTemplate testRt = new RestTemplate();
        SimpleClientHttpRequestFactory testFactory = new SimpleClientHttpRequestFactory();
        testFactory.setConnectTimeout(5000);
        testFactory.setReadTimeout(5000);
        testRt.setRequestFactory(testFactory);

        String healthUrl = pythonEngineUrl + "/health";
        long startTime = System.currentTimeMillis();
        boolean success = false;
        long elapsed = -1;
        String httpStatus = "N/A";
        String responseBody = "N/A";
        String errorMessage = "";

        try {
            ResponseEntity<Map> response = testRt.getForEntity(healthUrl, Map.class);
            elapsed = System.currentTimeMillis() - startTime;
            success = true;
            httpStatus = response.getStatusCode().toString();
            if (response.getBody() != null) {
                responseBody = response.getBody().toString();
            }
            System.out.println("Connection test: SUCCESS");
        } catch (Exception e) {
            elapsed = System.currentTimeMillis() - startTime;
            success = false;
            errorMessage = e.getMessage();
            System.out.println("Connection test: FAILED - " + errorMessage);
        }

        // Save results to scratch/connectivity_results.json
        try {
            File scratchDir = new File("scratch");
            if (!scratchDir.exists()) {
                scratchDir.mkdirs();
            }
            
            // Format JSON string manually to avoid adding jackson dependencies in test scope if not needed
            String json = String.format(
                "{\n" +
                "  \"python_url\": \"%s\",\n" +
                "  \"connect_timeout_ms\": %d,\n" +
                "  \"read_timeout_ms\": %d,\n" +
                "  \"retry_attempts\": %d,\n" +
                "  \"connection_success\": %b,\n" +
                "  \"response_time_ms\": %d,\n" +
                "  \"http_status\": \"%s\",\n" +
                "  \"response_body\": \"%s\",\n" +
                "  \"error_message\": \"%s\"\n" +
                "}",
                pythonEngineUrl,
                connectTimeout,
                readTimeout,
                maxAttempts,
                success,
                elapsed,
                httpStatus.replace("\"", "\\\""),
                responseBody.replace("\"", "\\\""),
                errorMessage.replace("\"", "\\\"")
            );
            
            FileWriter writer = new FileWriter("scratch/connectivity_results.json");
            writer.write(json);
            writer.close();
            System.out.println("Saved connectivity results to scratch/connectivity_results.json");
        } catch (Exception ex) {
            System.err.println("Failed to save connectivity results: " + ex.getMessage());
        }
    }
}
