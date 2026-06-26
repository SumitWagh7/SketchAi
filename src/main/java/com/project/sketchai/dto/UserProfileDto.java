package com.project.sketchai.dto;

import lombok.Builder;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.UUID;

@Data
@Builder
public class UserProfileDto {
    private UUID id;
    private String username;
    private String email;
    private String phone;
    private LocalDateTime createdAt;
}
