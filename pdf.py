import os
import sys
import json
import time
import logging
import pymupdf
import re

from models import (
    JSONResume,
    Basics,
    Work,
    Education,
    Skill,
    Project,
    Award,
    BasicsSection,
    WorkSection,
    EducationSection,
    SkillsSection,
    ProjectsSection,
    AwardsSection,
)
from llm_utils import initialize_llm_provider, extract_json_from_response
from pymupdf_rag import to_markdown
from typing import List, Optional, Dict, Any
from prompt import (
    DEFAULT_MODEL,
    MODEL_PARAMETERS,
    MODEL_PROVIDER_MAPPING,
    GEMINI_API_KEY,
)
from prompts.template_manager import TemplateManager
from transform import transform_parsed_data

logger = logging.getLogger(__name__)


def robust_json_parse(text: str) -> dict:
    """
    Robust JSON parser that handles truncated/incomplete JSON from LLMs.
    Tries multiple strategies to salvage partial JSON output.
    """
    # Strategy 1: Clean markdown code fences
    cleaned = text.strip()
    cleaned = re.sub(r'```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*', '', cleaned)
    
    # Strategy 2: Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Fix unterminated strings by closing quotes on lines
    lines = cleaned.split('\n')
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.count('"') % 2 != 0 and ':' in stripped:
            stripped = stripped + '"'
        fixed_lines.append(stripped)
    fixed = '\n'.join(fixed_lines)
    
    # Strategy 4: Try to find the last complete JSON structure
    for end_pattern in ['}]', '"]', '"}', ']', '}']:
        last_idx = fixed.rfind(end_pattern)
        if last_idx != -1:
            try:
                candidate = fixed[:last_idx + len(end_pattern)]
                if candidate.count('{') == candidate.count('}') and \
                   candidate.count('[') == candidate.count(']'):
                    return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    
    # Strategy 5: Try adding missing closing brackets
    for suffix in ['}]}', '}]', '"}', ']', '}']:
        try:
            candidate = fixed.rstrip().rstrip(',') + suffix
            if candidate.count('{') == candidate.count('}') and \
               candidate.count('[') == candidate.count(']'):
                return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    # If all strategies fail, raise the original error
    raise json.JSONDecodeError("Could not parse LLM response as JSON", cleaned, 0)


class PDFHandler:
    def __init__(self):
        self.template_manager = TemplateManager()
        self._initialize_llm_provider()

    def _initialize_llm_provider(self):
        """Initialize the appropriate LLM provider based on the model."""
        self.provider = initialize_llm_provider(DEFAULT_MODEL)

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            with pymupdf.open(pdf_path) as doc:
                pages = range(doc.page_count)
                resume_text = to_markdown(
                    doc,
                    pages=pages,
                )
                logger.debug(
                    f"Extracted text from PDF: {len(resume_text) if resume_text else 0} characters"
                )
                return resume_text
        except Exception as e:
            logger.error(f"An error occurred while reading the PDF: {e}")
            return None

    def _call_llm_for_section(
        self, section_name: str, text_content: str, prompt: str, return_model=None
    ) -> Optional[Dict]:
        try:
            start_time = time.time()
            logger.debug(
                f"🔄 Extracting {section_name} section using {DEFAULT_MODEL}..."
            )

            model_params = MODEL_PARAMETERS.get(
                DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9}
            )

            section_system_message = self.template_manager.render_template(
                "system_message", section_name_param=section_name
            )
            if not section_system_message:
                section_system_message = f"You are a resume parser. Extract ONLY the {section_name} section as valid JSON. No markdown, no commentary."

            chat_params = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": section_system_message},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "temperature": model_params.get("temperature", 0.1),
                    "top_p": model_params.get("top_p", 0.9),
                    "stream": False,
                },
            }

            kwargs = {}
            if return_model:
                kwargs["format"] = return_model.model_json_schema()

            # Use the appropriate provider to make the API call
            response = self.provider.chat(**chat_params, **kwargs)

            response_text = response["message"]["content"]

            # Clean and parse the JSON response
            response_text = extract_json_from_response(response_text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            if json_start != -1 and json_end != -1:
                response_text = response_text[json_start : json_end + 1]

            # Use robust JSON parser instead of direct json.loads
            parsed_data = robust_json_parse(response_text)
            logger.debug(f"✅ Successfully extracted {section_name} section")

            transformed_data = transform_parsed_data(parsed_data)
            end_time = time.time()
            total_time = end_time - start_time
            logger.debug(
                f"⏱️ Total time for {section_name} section extraction: {total_time:.2f} seconds"
            )

            return transformed_data

        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON for {section_name} section: {e}")
            logger.error(f"Raw response (first 300 chars): {response_text[:300]}")
            return None

        except Exception as e:
            logger.error(f"❌ Error calling LLM for {section_name} section: {e}")
            return None

    def extract_basics_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_template(
            "basics", text_content=resume_text
        )
        if not prompt:
            logger.error("❌ Failed to render basics template")
            return None
        return self._call_llm_for_section("basics", resume_text, prompt, BasicsSection)

    def extract_work_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_template("work", text_content=resume_text)
        if not prompt:
            logger.error("❌ Failed to render work template")
            return None
        return self._call_llm_for_section("work", resume_text, prompt, WorkSection)

    def extract_education_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_template(
            "education", text_content=resume_text
        )
        if not prompt:
            logger.error("❌ Failed to render education template")
            return None
        return self._call_llm_for_section(
            "education", resume_text, prompt, EducationSection
        )

    def extract_skills_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_template(
            "skills", text_content=resume_text
        )
        if not prompt:
            logger.error("❌ Failed to render skills template")
            return None
        return self._call_llm_for_section("skills", resume_text, prompt, SkillsSection)

    def extract_projects_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_template(
            "projects", text_content=resume_text
        )
        if not prompt:
            logger.error("❌ Failed to render projects template")
            return None
        return self._call_llm_for_section(
            "projects", resume_text, prompt, ProjectsSection
        )

    def extract_awards_section(self, resume_text: str) -> Optional[Dict]:
        prompt = self.template_manager.render_template(
            "awards", text_content=resume_text
        )
        if not prompt:
            logger.error("❌ Failed to render awards template")
            return None
        return self._call_llm_for_section("awards", resume_text, prompt, AwardsSection)

    def extract_json_from_text(self, resume_text: str) -> Optional[JSONResume]:
        try:
            return self._extract_all_sections_separately(resume_text)
        except Exception as e:
            logger.error(f"Error extracting resume data: {e}")
            return None

    def extract_json_from_pdf(self, pdf_path: str) -> Optional[JSONResume]:
        try:
            logger.debug(f"📄 Extracting text from PDF: {pdf_path}")
            text_content = self.extract_text_from_pdf(pdf_path)

            if not text_content:
                logger.error("❌ Failed to extract text from PDF")
                return None

            logger.debug(
                f"✅ Successfully extracted {len(text_content)} characters from PDF"
            )

            logger.debug("🔄 Extracting all sections separately...")
            return self._extract_all_sections_separately(text_content)

        except Exception as e:
            logger.error(f"❌ Error during PDF to JSON extraction: {e}")
            return None

    def _extract_section_data(
        self, text_content: str, section_name: str, return_model=None
    ) -> Optional[Dict]:
        section_extractors = {
            "basics": self.extract_basics_section,
            "work": self.extract_work_section,
            "education": self.extract_education_section,
            "skills": self.extract_skills_section,
            "projects": self.extract_projects_section,
            "awards": self.extract_awards_section,
        }

        if section_name not in section_extractors:
            logger.error(f"❌ Invalid section name: {section_name}")
            return None

        # Try once normally
        result = section_extractors[section_name](text_content)
        if result:
            return result
        
        # If first attempt failed, retry once (small models can be flaky)
        logger.warning(f"⚠️ First attempt for {section_name} failed, retrying once...")
        time.sleep(1)  # Brief pause before retry
        result = section_extractors[section_name](text_content)
        if result:
            logger.info(f"✅ Retry succeeded for {section_name}")
        return result

    def _extract_all_sections_separately(
        self, text_content: str
    ) -> Optional[JSONResume]:
        start_time = time.time()

        sections = ["basics", "work", "education", "skills", "projects", "awards"]

        complete_resume = {
            "basics": None,
            "work": None,
            "volunteer": None,
            "education": None,
            "awards": None,
            "certificates": None,
            "publications": None,
            "skills": None,
            "languages": None,
            "interests": None,
            "references": None,
            "projects": None,
            "meta": None,
        }

        failed_sections = []
        
        for section_name in sections:
            section_data = self._extract_section_data(text_content, section_name)

            if section_data:
                complete_resume.update(section_data)
                logger.debug(f"✅ Successfully extracted {section_name} section")
            else:
                failed_sections.append(section_name)
                logger.warning(
                    f"⚠️ Failed to extract {section_name} section. Continuing with remaining sections."
                )

        # Check if we got at least basics (minimum viable resume)
        if complete_resume.get("basics") is None:
            # Check if we got ANY data at all
            has_any_data = any(
                complete_resume.get(key) is not None
                for key in ["work", "education", "skills", "projects"]
            )
            if not has_any_data:
                logger.error(
                    "❌ Failed to extract any meaningful sections. Aborting."
                )
                return None
            else:
                logger.warning(
                    "⚠️ Basics section failed but other sections extracted. Creating minimal resume."
                )

        if failed_sections:
            logger.warning(
                f"⚠️ The following sections could not be extracted: {', '.join(failed_sections)}"
            )

        try:
            # Convert basics dict to Pydantic model if needed
            if complete_resume.get("basics") and isinstance(
                complete_resume["basics"], dict
            ):
                try:
                    complete_resume["basics"] = Basics(**complete_resume["basics"])
                except Exception as e:
                    logger.warning(f"⚠️ Could not validate basics model: {e}")

            json_resume = JSONResume(**complete_resume)
            
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(
                f"⏱️ Total time for separate section extraction: {total_time:.2f} seconds"
            )

            return json_resume

        except Exception as e:
            logger.error(f"❌ Error creating JSONResume model: {e}")
            # Try to return partial data even if model creation fails
            try:
                return JSONResume(**{k: v for k, v in complete_resume.items() if v is not None})
            except Exception:
                return None