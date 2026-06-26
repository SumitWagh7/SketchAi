package com.project.sketchai.repository;

import com.project.sketchai.model.AiRequest;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface AiRequestRepository extends JpaRepository<AiRequest, UUID> {
    List<AiRequest> findByUserIdOrderByRequestedAtDesc(UUID userId);
}
