#!/usr/bin/env python3
"""
Compilation Status Utilities
Provides a simplified compilation success indicator for Java build logs.
"""

import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CompilationResult:
    """Summarized compilation result for a single build log."""
    project: str
    build_tool: str  # gradle, maven, or unknown
    is_compile_ok: bool
    error_count: int = 0
    
    def to_dict(self):
        return asdict(self)


class CompilationAnalyzer:
    """Analyze build logs to determine compilation success.
    
    This analyzer focuses on detecting compilation errors only,
    distinguishing them from test failures or other build issues.
    """
    
    def __init__(self):
        self.error_pattern = re.compile(r'\[ERROR\]|\berror\b', re.IGNORECASE)
        self.ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\[\[?[0-9;]*m')
        
        # Gradle compile task failure pattern
        # Matches: "> Task :xxx:compileJava FAILED" or "> Task :compileTestJava FAILED"
        self.gradle_compile_fail_pattern = re.compile(
            r'> Task [:\w-]*:?compile(?:Test)?(?:Java|Groovy|Kotlin|Scala)\s+FAILED',
            re.IGNORECASE
        )
        
        # Common javac error patterns (fallback detection)
        self.javac_error_patterns = [
            re.compile(r'error:\s*cannot find symbol', re.IGNORECASE),
            re.compile(r'error:\s*package .+ does not exist', re.IGNORECASE),
            re.compile(r'error:\s*incompatible types', re.IGNORECASE),
            re.compile(r'error:\s*method .+ cannot be applied', re.IGNORECASE),
            re.compile(r'error:\s*class .+ is public, should be declared', re.IGNORECASE),
            re.compile(r'error:\s*\w+ has private access', re.IGNORECASE),
            re.compile(r'error:\s*unreported exception', re.IGNORECASE),
            re.compile(r'error:\s*illegal start of', re.IGNORECASE),
            re.compile(r'error:\s*not a statement', re.IGNORECASE),
            re.compile(r'error:\s*\';\' expected', re.IGNORECASE),
        ]
    
    def _remove_ansi_codes(self, text: str) -> str:
        """Strip ANSI escape sequences from log text."""
        return self.ansi_escape.sub('', text)
    
    def _detect_build_tool(self, log: str) -> str:
        """Auto-detect the build tool from the log snippet."""
        log_lower = log.lower()
        
        if './gradlew' in log_lower or 'gradle' in log_lower or '> task :' in log_lower:
            return 'gradle'
        if 'mvn ' in log_lower or 'maven' in log_lower or '[info]' in log_lower:
            return 'maven'
        return 'unknown'
    
    def _is_compile_ok(self, clean_log: str, build_tool: str) -> bool:
        """
        Check if compilation succeeded.
        
        Only detects compile-phase errors, ignores test failures.
        This provides a more accurate compilation success indicator
        compared to checking BUILD FAILED which includes test failures.
        
        Args:
            clean_log: Log content with ANSI codes removed
            build_tool: Detected build tool (gradle, maven, or unknown)
            
        Returns:
            True if compilation succeeded, False if compilation errors found
        """
        upper_log = clean_log.upper()
        
        # Maven: check for explicit COMPILATION ERROR marker
        if build_tool == 'maven':
            if 'COMPILATION ERROR' in upper_log:
                return False
        
        # Gradle: check for compileJava/compileTestJava task failures
        elif build_tool == 'gradle':
            if self.gradle_compile_fail_pattern.search(clean_log):
                return False
            # Also check for "Compilation failed" message in Gradle
            if 'COMPILATION FAILED' in upper_log:
                return False
        
        # Fallback: check for common javac error patterns
        # These patterns indicate actual compilation errors regardless of build tool
        for pattern in self.javac_error_patterns:
            if pattern.search(clean_log):
                return False
        
        return True
    
    def _count_errors(self, clean_log: str) -> int:
        """Count the number of lines that mention errors."""
        return sum(1 for line in clean_log.splitlines() if self.error_pattern.search(line))
    
    def analyze_log_file(self, log_path: Path, project_name: Optional[str] = None) -> CompilationResult:
        """
        Analyze a single log file.
        
        Args:
            log_path: Path to the log file
            project_name: Optional project identifier
            
        Returns:
            CompilationResult with simplified success info
        """
        if not project_name:
            parts = log_path.parts
            project_name = '/'.join(parts[-5:-3]) if len(parts) >= 5 else log_path.stem
        
        try:
            log_content = log_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return CompilationResult(
                project=project_name,
                build_tool='error',
                is_compile_ok=False,
                error_count=-1
            )
        
        return self.analyze_log_content(log_content, project_name)
    
    def analyze_log_content(self, log_content: str, project_name: str) -> CompilationResult:
        """
        Analyze log content directly.
        
        Args:
            log_content: Log file content as string
            project_name: Project identifier
            
        Returns:
            CompilationResult with simplified success info
        """
        clean_log = self._remove_ansi_codes(log_content)
        build_tool = self._detect_build_tool(clean_log)
        is_compile_ok = self._is_compile_ok(clean_log, build_tool)
        error_count = self._count_errors(clean_log)
        
        return CompilationResult(
            project=project_name,
            build_tool=build_tool,
            is_compile_ok=is_compile_ok,
            error_count=error_count
        )


# Convenience helpers
def analyze_compilation_log(log_path: Path, project_name: Optional[str] = None) -> CompilationResult:
    analyzer = CompilationAnalyzer()
    return analyzer.analyze_log_file(log_path, project_name)


def analyze_compilation_content(log_content: str, project_name: str) -> CompilationResult:
    analyzer = CompilationAnalyzer()
    return analyzer.analyze_log_content(log_content, project_name)