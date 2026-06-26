package com.project.sketchai.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "ai_requests")
public class AiRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private String actionType; // e.g., "GHIBLI", "CAPTION", "SUGGESTION"

    @Column(columnDefinition = "TEXT")
    private String promptUsed;

    @Column(columnDefinition = "TEXT")
    private String resultUrlOrText;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime requestedAt;
}
