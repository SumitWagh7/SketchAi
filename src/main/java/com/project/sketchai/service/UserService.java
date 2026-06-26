package com.project.sketchai.service;

import com.project.sketchai.dto.UserProfileDto;
import com.project.sketchai.model.User;
import com.project.sketchai.repository.ActivityLogRepository;
import com.project.sketchai.repository.AiRequestRepository;
import com.project.sketchai.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final AiRequestRepository aiRequestRepository;
    private final ActivityLogRepository activityLogRepository;

    public User getCurrentUser() {
        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        if (principal instanceof UserDetails) {
            String email = ((UserDetails) principal).getUsername();
            return userRepository.findByEmail(email)
                    .orElseThrow(() -> new RuntimeException("User not found"));
        }
        throw new RuntimeException("Not authenticated");
    }

    public UserProfileDto getUserProfile() {
        User user = getCurrentUser();
        return UserProfileDto.builder()
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .phone(user.getPhone())
                .createdAt(user.getCreatedAt())
                .build();
    }

    public Map<String, Object> getDashboardStats() {
        User user = getCurrentUser();
        
        long totalAiRequests = aiRequestRepository.findByUserIdOrderByRequestedAtDesc(user.getId()).size();
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalRequests", totalAiRequests);
        stats.put("recentActivities", activityLogRepository.findByUserIdOrderByTimestampDesc(user.getId()));
        
        return stats;
    }
}
