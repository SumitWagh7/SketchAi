package com.project.sketchai.dto;

import lombok.Data;

@Data
public class AiRequestDto {
    private String prompt;
    private String image; // base64
    private String mask;  // base64 for inpainting
    private Double strength;
    private Integer upscaleFactor;
    private String mode;
}
