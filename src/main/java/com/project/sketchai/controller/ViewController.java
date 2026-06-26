package com.project.sketchai.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class ViewController {

    @GetMapping({
        "/",
        "/login",
        "/dashboard",
        "/ghibli",
        "/background-removal",
        "/background-replacement",
        "/object-removal",
        "/upscale",
        "/profile",
        "/settings"
    })
    public String index() {
        return "forward:/index.html";
    }

    @GetMapping("/login.html")
    public String redirectToLogin() {
        return "redirect:/login";
    }

    @GetMapping("/dashboard.html")
    public String redirectToDashboard() {
        return "redirect:/dashboard";
    }
}
