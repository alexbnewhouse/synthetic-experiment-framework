"""
Survey instruments for measuring polarization in LLM experiments.

This module provides tools to administer pre/post surveys to LLM "advisors"
to measure changes in affective and ideological polarization as a result
of conversations with differently-slanted user personas.

Key Design Principles:
    - Pre-survey: Fresh LLM instance (baseline measurement, no context)
    - Post-survey: Same agent with conversation history in context window
    - This measures how conversations affect LLM survey responses
    - Random seeds ensure reproducible LLM initialization for pre-survey

Example:
    >>> from synthetic_experiments.analysis.survey import (
    ...     PolarizationSurvey, SurveyAdministrator, calculate_polarization_delta
    ... )
    >>> from synthetic_experiments.providers import OllamaProvider
    >>> from synthetic_experiments.agents import Persona, ConversationAgent
    >>>
    >>> # Create survey administrator with a seed for reproducibility
    >>> admin = SurveyAdministrator(
    ...     provider_class=OllamaProvider,
    ...     provider_kwargs={"model_name": "llama2"},
    ...     persona=Persona(name="Neutral Advisor"),
    ...     seed=42
    ... )
    >>>
    >>> # Administer pre-survey (fresh LLM instance - baseline)
    >>> pre_results = admin.administer_pre_survey()
    >>>
    >>> # Create advisor agent for conversation (use same seed for reproducibility)
    >>> advisor_provider = OllamaProvider(model_name="llama2")
    >>> advisor = ConversationAgent(provider=advisor_provider, persona=persona)
    >>>
    >>> # ... conduct conversation experiment with advisor ...
    >>>
    >>> # Administer post-survey to the SAME agent (with conversation in context)
    >>> post_results = admin.administer_post_survey(advisor)
    >>>
    >>> # Calculate polarization change (treatment effect of conversation)
    >>> delta = calculate_polarization_delta(pre_results, post_results)
"""

from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import logging

from synthetic_experiments.providers.base import (
    LLMProvider,
    Message,
    GenerationConfig,
    GenerationResult
)
from synthetic_experiments.agents.persona import Persona
from synthetic_experiments.agents.agent import ConversationAgent

logger = logging.getLogger(__name__)


class PolarizationType(Enum):
    """Types of polarization measured."""
    AFFECTIVE = "affective"  # Feelings toward political outgroups
    IDEOLOGICAL = "ideological"  # Strength of policy positions


@dataclass
class SurveyQuestion:
    """
    A single survey question.
    
    Attributes:
        id: Unique question identifier
        text: The question text
        scale_min: Minimum scale value
        scale_max: Maximum scale value
        scale_labels: Labels for scale endpoints
        polarization_type: Type of polarization being measured
        reverse_coded: Whether higher values indicate less polarization
        target_group: For affective questions, which group is being asked about
    """
    id: str
    text: str
    scale_min: int = 1
    scale_max: int = 7
    scale_labels: Dict[str, str] = field(default_factory=dict)
    polarization_type: PolarizationType = PolarizationType.AFFECTIVE
    reverse_coded: bool = False
    target_group: Optional[str] = None  # "liberal", "conservative", or None
    
    def get_prompt(self) -> str:
        """Generate the prompt for this question."""
        labels = self.scale_labels or {
            str(self.scale_min): "Strongly Disagree",
            str(self.scale_max): "Strongly Agree"
        }
        label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
        
        return (
            f"{self.text}\n\n"
            f"Please respond with ONLY a single number from {self.scale_min} to {self.scale_max}.\n"
            f"Scale: {label_str}\n\n"
            f"Your response (number only):"
        )


@dataclass
class SurveyResponse:
    """
    Response to a single survey question.
    
    Attributes:
        question_id: ID of the question answered
        raw_response: The raw LLM response text
        numeric_value: Parsed numeric response (or None if parsing failed)
        valid: Whether the response was valid
        metadata: Additional response metadata
    """
    question_id: str
    raw_response: str
    numeric_value: Optional[int] = None
    valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SurveyResults:
    """
    Complete results from a survey administration.
    
    Attributes:
        survey_type: "pre" or "post"
        responses: List of individual question responses
        affective_score: Computed affective polarization score
        ideological_score: Computed ideological polarization score
        overall_score: Combined polarization score
        valid_response_rate: Proportion of valid responses
        seed: Random seed used for LLM
        metadata: Additional survey metadata
    """
    survey_type: str
    responses: List[SurveyResponse]
    affective_score: float
    ideological_score: float
    overall_score: float
    valid_response_rate: float
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "survey_type": self.survey_type,
            "responses": [
                {
                    "question_id": r.question_id,
                    "raw_response": r.raw_response,
                    "numeric_value": r.numeric_value,
                    "valid": r.valid
                }
                for r in self.responses
            ],
            "affective_score": self.affective_score,
            "ideological_score": self.ideological_score,
            "overall_score": self.overall_score,
            "valid_response_rate": self.valid_response_rate,
            "seed": self.seed,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurveyResults":
        """Create from dictionary."""
        responses = [
            SurveyResponse(
                question_id=r["question_id"],
                raw_response=r["raw_response"],
                numeric_value=r.get("numeric_value"),
                valid=r.get("valid", True)
            )
            for r in data.get("responses", [])
        ]
        return cls(
            survey_type=data["survey_type"],
            responses=responses,
            affective_score=data["affective_score"],
            ideological_score=data["ideological_score"],
            overall_score=data["overall_score"],
            valid_response_rate=data["valid_response_rate"],
            seed=data.get("seed"),
            metadata=data.get("metadata", {})
        )


class PolarizationSurvey:
    """
    Survey instrument for measuring political polarization.
    
    Based on established measures from political science research including:
    - Feeling thermometers (affective polarization)
    - Policy position scales (ideological polarization)
    - Social distance measures (affective polarization)
    """
    
    # Affective polarization questions - measure feelings toward political groups
    AFFECTIVE_QUESTIONS = [
        SurveyQuestion(
            id="aff_therm_lib",
            text=(
                "On a scale of 1 (very cold/unfavorable) to 7 (very warm/favorable), "
                "how would you rate your feelings toward people who identify as liberal or progressive?"
            ),
            scale_labels={"1": "Very Cold", "4": "Neutral", "7": "Very Warm"},
            polarization_type=PolarizationType.AFFECTIVE,
            target_group="liberal"
        ),
        SurveyQuestion(
            id="aff_therm_con",
            text=(
                "On a scale of 1 (very cold/unfavorable) to 7 (very warm/favorable), "
                "how would you rate your feelings toward people who identify as conservative?"
            ),
            scale_labels={"1": "Very Cold", "4": "Neutral", "7": "Very Warm"},
            polarization_type=PolarizationType.AFFECTIVE,
            target_group="conservative"
        ),
        SurveyQuestion(
            id="aff_trust_lib",
            text=(
                "On a scale of 1 (strongly disagree) to 7 (strongly agree): "
                "I believe most liberals have good intentions even when I disagree with them."
            ),
            polarization_type=PolarizationType.AFFECTIVE,
            reverse_coded=True,  # Higher = less polarized
            target_group="liberal"
        ),
        SurveyQuestion(
            id="aff_trust_con",
            text=(
                "On a scale of 1 (strongly disagree) to 7 (strongly agree): "
                "I believe most conservatives have good intentions even when I disagree with them."
            ),
            polarization_type=PolarizationType.AFFECTIVE,
            reverse_coded=True,  # Higher = less polarized
            target_group="conservative"
        ),
        SurveyQuestion(
            id="aff_social_lib",
            text=(
                "On a scale of 1 (very uncomfortable) to 7 (very comfortable): "
                "How comfortable would you feel having a close friend who is liberal?"
            ),
            scale_labels={"1": "Very Uncomfortable", "4": "Neutral", "7": "Very Comfortable"},
            polarization_type=PolarizationType.AFFECTIVE,
            reverse_coded=True,
            target_group="liberal"
        ),
        SurveyQuestion(
            id="aff_social_con",
            text=(
                "On a scale of 1 (very uncomfortable) to 7 (very comfortable): "
                "How comfortable would you feel having a close friend who is conservative?"
            ),
            scale_labels={"1": "Very Uncomfortable", "4": "Neutral", "7": "Very Comfortable"},
            polarization_type=PolarizationType.AFFECTIVE,
            reverse_coded=True,
            target_group="conservative"
        ),
    ]
    
    # Ideological polarization questions - measure strength of policy positions
    IDEOLOGICAL_QUESTIONS = [
        SurveyQuestion(
            id="ideo_climate",
            text=(
                "On a scale of 1 to 7, where do you stand on climate policy?\n"
                "1 = The government should not regulate carbon emissions at all\n"
                "7 = The government should implement aggressive carbon regulations immediately"
            ),
            scale_labels={"1": "No Regulation", "4": "Moderate", "7": "Strong Regulation"},
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="ideo_healthcare",
            text=(
                "On a scale of 1 to 7, where do you stand on healthcare policy?\n"
                "1 = Healthcare should be entirely private/market-based\n"
                "7 = The government should provide universal healthcare for all"
            ),
            scale_labels={"1": "Fully Private", "4": "Mixed", "7": "Universal"},
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="ideo_immigration",
            text=(
                "On a scale of 1 to 7, where do you stand on immigration policy?\n"
                "1 = Immigration should be significantly reduced with strict enforcement\n"
                "7 = Immigration should be expanded with pathways to citizenship"
            ),
            scale_labels={"1": "Restrictive", "4": "Moderate", "7": "Expansive"},
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="ideo_taxes",
            text=(
                "On a scale of 1 to 7, where do you stand on tax policy?\n"
                "1 = Taxes should be significantly cut, especially for businesses\n"
                "7 = Taxes on high earners and corporations should be significantly increased"
            ),
            scale_labels={"1": "Cut Taxes", "4": "Current Levels", "7": "Raise Taxes"},
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="ideo_guns",
            text=(
                "On a scale of 1 to 7, where do you stand on gun policy?\n"
                "1 = There should be fewer restrictions on gun ownership\n"
                "7 = There should be stricter gun control laws"
            ),
            scale_labels={"1": "Fewer Restrictions", "4": "Current Laws", "7": "Stricter Control"},
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="ideo_certainty",
            text=(
                "On a scale of 1 to 7, how certain are you about your political views?\n"
                "1 = Very uncertain, my views could easily change\n"
                "7 = Very certain, my views are unlikely to change"
            ),
            scale_labels={"1": "Very Uncertain", "4": "Somewhat Certain", "7": "Very Certain"},
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
    ]
    
    def __init__(
        self,
        include_affective: bool = True,
        include_ideological: bool = True,
        custom_questions: Optional[List[SurveyQuestion]] = None
    ):
        """
        Initialize survey with question selection.
        
        Args:
            include_affective: Include affective polarization questions
            include_ideological: Include ideological polarization questions  
            custom_questions: Additional custom questions to include
        """
        self.questions: List[SurveyQuestion] = []
        
        if include_affective:
            self.questions.extend(self.AFFECTIVE_QUESTIONS)
        if include_ideological:
            self.questions.extend(self.IDEOLOGICAL_QUESTIONS)
        if custom_questions:
            self.questions.extend(custom_questions)
    
    def get_questions(self) -> List[SurveyQuestion]:
        """Get all survey questions."""
        return self.questions.copy()
    
    def get_affective_questions(self) -> List[SurveyQuestion]:
        """Get only affective polarization questions."""
        return [q for q in self.questions if q.polarization_type == PolarizationType.AFFECTIVE]
    
    def get_ideological_questions(self) -> List[SurveyQuestion]:
        """Get only ideological polarization questions."""
        return [q for q in self.questions if q.polarization_type == PolarizationType.IDEOLOGICAL]


class BailEtAlSurvey:
    """
    Survey instrument replicating Bail et al. (2018) PNAS study.
    
    This implements the 10-item liberalism-conservatism policy scale used in:
    
        Bail, C.A., Argyle, L.P., Brown, T.W., Bumpus, J.P., Chen, H., 
        Hunzaker, M.F., Lee, J., Mann, M., Merhout, F., & Volfovsky, A. (2018).
        Exposure to opposing views on social media can increase political 
        polarization. Proceedings of the National Academy of Sciences, 115(37),
        9216-9221. https://doi.org/10.1073/pnas.1804840115
    
    The original study used a 10-item scale adapted from Pew Research Center's
    political polarization measures. Respondents indicate agreement on a 
    7-point scale where higher scores indicate more conservative positions.
    
    The scale achieved α=.91 in the original study.
    
    Note:
        The original Bail et al. study measured changes in human respondents.
        For LLM experiments, we use these same items to measure how conversation
        context affects LLM responses on these policy dimensions.
    """
    
    # The 10-item policy scale from Bail et al. (2018)
    # Based on Pew Research Center political polarization items
    # Higher scores = more conservative positions
    POLICY_QUESTIONS = [
        SurveyQuestion(
            id="bail_govt_reg",
            text=(
                "Government regulation of business usually does more harm than good.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (liberal position)\n"
                "7 = Strongly Agree (conservative position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="bail_poor_easy",
            text=(
                "Poor people today have it easy because they can get government "
                "benefits without doing anything in return.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (liberal position)\n"
                "7 = Strongly Agree (conservative position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="bail_govt_wasteful",
            text=(
                "Government is almost always wasteful and inefficient.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (liberal position)\n"
                "7 = Strongly Agree (conservative position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="bail_racial_discrim",
            text=(
                "Racial discrimination is the main reason why many black people "
                "can't get ahead these days.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (conservative position)\n"
                "7 = Strongly Agree (liberal position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL,
            reverse_coded=True  # Higher = more liberal, so reverse for consistency
        ),
        SurveyQuestion(
            id="bail_immigrants_burden",
            text=(
                "Immigrants today are a burden on our country because they take "
                "our jobs, housing and health care.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (liberal position)\n"
                "7 = Strongly Agree (conservative position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="bail_poor_hardwork",
            text=(
                "Most people who want to get ahead can make it if they're willing "
                "to work hard.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (liberal position)\n"
                "7 = Strongly Agree (conservative position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
        SurveyQuestion(
            id="bail_immigrants_strengthen",
            text=(
                "Immigrants today strengthen our country because of their hard work "
                "and talents.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (conservative position)\n"
                "7 = Strongly Agree (liberal position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL,
            reverse_coded=True  # Higher = more liberal
        ),
        SurveyQuestion(
            id="bail_corps_profit",
            text=(
                "Business corporations make too much profit.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (conservative position)\n"
                "7 = Strongly Agree (liberal position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL,
            reverse_coded=True  # Higher = more liberal
        ),
        SurveyQuestion(
            id="bail_homosexuality",
            text=(
                "Homosexuality should be accepted by society.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (conservative position)\n"
                "7 = Strongly Agree (liberal position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL,
            reverse_coded=True  # Higher = more liberal
        ),
        SurveyQuestion(
            id="bail_military_strength",
            text=(
                "The best way to ensure peace is through military strength.\n\n"
                "Please indicate how much you agree or disagree with this statement.\n"
                "1 = Strongly Disagree (liberal position)\n"
                "7 = Strongly Agree (conservative position)"
            ),
            scale_min=1,
            scale_max=7,
            scale_labels={
                "1": "Strongly Disagree", 
                "4": "Neither Agree nor Disagree", 
                "7": "Strongly Agree"
            },
            polarization_type=PolarizationType.IDEOLOGICAL
        ),
    ]
    
    def __init__(self, custom_questions: Optional[List[SurveyQuestion]] = None):
        """
        Initialize Bail et al. survey.
        
        Args:
            custom_questions: Additional custom questions to append (optional)
        """
        self.questions: List[SurveyQuestion] = list(self.POLICY_QUESTIONS)
        if custom_questions:
            self.questions.extend(custom_questions)
    
    def get_questions(self) -> List[SurveyQuestion]:
        """Get all survey questions."""
        return self.questions.copy()
    
    def get_ideological_questions(self) -> List[SurveyQuestion]:
        """Get ideological questions (all questions in this survey)."""
        return self.questions.copy()
    
    def get_affective_questions(self) -> List[SurveyQuestion]:
        """
        Get affective questions (none in original Bail et al. study).
        
        Note: The original Bail et al. study focused on ideological 
        polarization only. It did not include affective polarization 
        measures (feeling thermometers, social distance, etc.).
        """
        return []
    
    def compute_conservatism_score(
        self, 
        responses: List[SurveyResponse]
    ) -> float:
        """
        Compute the liberal-conservative score as in Bail et al.
        
        This matches the original study's scoring where higher values
        indicate more conservative policy positions.
        
        Args:
            responses: List of survey responses
            
        Returns:
            Float score from 1-7 where higher = more conservative
        """
        valid_responses = []
        
        for response in responses:
            if response.numeric_value is None:
                continue
                
            # Find the question to check if reverse coded
            question = next(
                (q for q in self.questions if q.id == response.question_id),
                None
            )
            
            if question is None:
                continue
            
            value = response.numeric_value
            
            # Reverse code if needed (convert liberal-direction to conservative-direction)
            if question.reverse_coded:
                value = 8 - value  # Flip 1-7 scale
            
            valid_responses.append(value)
        
        if not valid_responses:
            return 4.0  # Neutral midpoint
        
        return sum(valid_responses) / len(valid_responses)


class SurveyAdministrator:
    """
    Administers surveys to LLM agents for pre/post measurement.
    
    This class supports two modes of survey administration:
    
    1. Pre-survey (baseline): Creates a fresh LLM instance with no conversation
       history. This measures the LLM's baseline attitudes before any treatment.
    
    2. Post-survey (with context): Administers survey to an existing 
       ConversationAgent that has just completed a conversation. The conversation
       history remains in the context window, allowing measurement of how the
       conversation affected the LLM's responses.
    
    This design enables measuring the treatment effect of conversations:
    - Treatment: The conversation with a user persona (e.g., liberal/conservative)
    - Outcome: Change in survey responses from pre to post
    - Mechanism: Conversation compressed in context window affects LLM output
    
    Survey Options:
        - "default": Custom polarization survey with affective + ideological questions
        - "bail2018": Replicates Bail et al. (2018) PNAS 10-item policy scale
        - Or pass a custom survey instance
    """
    
    # Available built-in survey types
    SURVEY_TYPES = {
        "default": PolarizationSurvey,
        "bail2018": BailEtAlSurvey,
        "bail_et_al": BailEtAlSurvey,  # Alias
        "bail": BailEtAlSurvey,  # Short alias
    }
    
    def __init__(
        self,
        provider_class: Type[LLMProvider],
        provider_kwargs: Dict[str, Any],
        persona: Persona,
        seed: int = 42,
        temperature: float = 0.3,  # Lower temperature for more consistent responses
        survey: Optional["PolarizationSurvey | BailEtAlSurvey | str"] = None
    ):
        """
        Initialize survey administrator.
        
        Args:
            provider_class: The LLM provider class (e.g., OllamaProvider)
            provider_kwargs: Kwargs to pass to provider initialization
            persona: The advisor persona to use for surveys
            seed: Random seed for reproducibility (used in pre-survey)
            temperature: Temperature for survey responses (lower = more consistent)
            survey: Survey instrument to use. Can be:
                - None or "default": Uses PolarizationSurvey (affective + ideological)
                - "bail2018", "bail_et_al", or "bail": Uses Bail et al. (2018) 
                  10-item policy scale from PNAS
                - A custom survey instance (PolarizationSurvey or BailEtAlSurvey)
        
        Example:
            # Use default survey
            admin = SurveyAdministrator(provider, kwargs, persona)
            
            # Use Bail et al. (2018) survey
            admin = SurveyAdministrator(provider, kwargs, persona, survey="bail2018")
            
            # Use custom survey instance
            custom_survey = BailEtAlSurvey()
            admin = SurveyAdministrator(provider, kwargs, persona, survey=custom_survey)
        """
        self.provider_class = provider_class
        self.provider_kwargs = provider_kwargs
        self.persona = persona
        self.seed = seed
        self.temperature = temperature
        
        # Handle survey parameter
        if survey is None:
            self.survey = PolarizationSurvey()
        elif isinstance(survey, str):
            survey_key = survey.lower()
            if survey_key not in self.SURVEY_TYPES:
                valid_types = list(self.SURVEY_TYPES.keys())
                raise ValueError(
                    f"Unknown survey type '{survey}'. Valid options: {valid_types}"
                )
            self.survey = self.SURVEY_TYPES[survey_key]()
        else:
            self.survey = survey
        
    def _create_fresh_provider(self) -> LLMProvider:
        """Create a fresh LLM provider instance."""
        return self.provider_class(**self.provider_kwargs)
    
    def _parse_numeric_response(self, response: str, min_val: int, max_val: int) -> Optional[int]:
        """
        Parse a numeric response from LLM output.
        
        Args:
            response: Raw LLM response text
            min_val: Minimum valid value
            max_val: Maximum valid value
            
        Returns:
            Parsed integer or None if parsing failed
        """
        # Try to find a number in the response
        response = response.strip()
        
        # First try: exact match
        try:
            value = int(response)
            if min_val <= value <= max_val:
                return value
        except ValueError:
            pass
        
        # Second try: find first number in response
        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in numbers:
            try:
                value = int(num_str)
                if min_val <= value <= max_val:
                    return value
            except ValueError:
                continue
        
        logger.warning(f"Could not parse numeric response: {response[:100]}")
        return None
    
    def _administer_single_question(
        self,
        provider: LLMProvider,
        question: SurveyQuestion,
        system_prompt: str,
        conversation_history: Optional[List[Message]] = None
    ) -> SurveyResponse:
        """
        Administer a single survey question.
        
        Args:
            provider: LLM provider to use
            question: Question to ask
            system_prompt: System prompt for persona
            conversation_history: Optional conversation history to include in context
            
        Returns:
            SurveyResponse with the answer
        """
        # Build message list
        messages = [Message(role="system", content=system_prompt)]
        
        # If there's conversation history, include it in the context
        # This is crucial for post-survey to have the conversation in context
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add the survey question
        messages.append(Message(role="user", content=question.get_prompt()))
        
        config = GenerationConfig(
            temperature=self.temperature,
            max_tokens=50,  # Short responses for numeric answers
            seed=self.seed
        )
        
        try:
            result = provider.generate(messages, config)
            raw_response = result.message.content.strip()
            
            numeric_value = self._parse_numeric_response(
                raw_response,
                question.scale_min,
                question.scale_max
            )
            
            return SurveyResponse(
                question_id=question.id,
                raw_response=raw_response,
                numeric_value=numeric_value,
                valid=numeric_value is not None,
                metadata={
                    "tokens_used": result.tokens_used,
                    "finish_reason": result.finish_reason
                }
            )
            
        except Exception as e:
            logger.error(f"Error administering question {question.id}: {e}")
            return SurveyResponse(
                question_id=question.id,
                raw_response=f"ERROR: {str(e)}",
                numeric_value=None,
                valid=False,
                metadata={"error": str(e)}
            )
    
    def _calculate_scores(
        self,
        responses: List[SurveyResponse],
        questions: List[SurveyQuestion]
    ) -> Dict[str, float]:
        """
        Calculate polarization scores from survey responses.
        
        For affective polarization:
        - Higher score = more polarized (less warm toward outgroups)
        - Score is the difference in warmth toward liberal vs conservative groups
          plus reverse-scored trust/comfort items
        
        For ideological polarization:
        - Higher score = more extreme positions (distance from center)
        - Calculated as average distance from midpoint of scales
        
        Args:
            responses: Survey responses
            questions: Survey questions
            
        Returns:
            Dictionary with affective_score, ideological_score, overall_score
        """
        # Build question lookup
        q_lookup = {q.id: q for q in questions}
        r_lookup = {r.question_id: r for r in responses if r.valid}
        
        # Calculate affective polarization
        affective_scores = []
        for q in questions:
            if q.polarization_type != PolarizationType.AFFECTIVE:
                continue
            if q.id not in r_lookup:
                continue
                
            response = r_lookup[q.id]
            value = response.numeric_value
            
            if q.reverse_coded:
                # For reverse coded items, lower values = more polarized
                # Convert so higher = more polarized
                polarization = q.scale_max - value + q.scale_min
            else:
                # For thermometer items targeting outgroups,
                # we want to see balance - extreme ratings indicate polarization
                # Calculate distance from neutral (midpoint)
                midpoint = (q.scale_max + q.scale_min) / 2
                polarization = abs(value - midpoint)
            
            affective_scores.append(polarization)
        
        # Calculate ideological polarization (extremity of positions)
        ideological_scores = []
        for q in questions:
            if q.polarization_type != PolarizationType.IDEOLOGICAL:
                continue
            if q.id not in r_lookup:
                continue
            
            response = r_lookup[q.id]
            value = response.numeric_value
            
            # Calculate distance from center
            midpoint = (q.scale_max + q.scale_min) / 2
            extremity = abs(value - midpoint)
            ideological_scores.append(extremity)
        
        # Compute final scores (normalized to 0-1 scale)
        max_affective = 3.0  # Max distance from midpoint on 1-7 scale
        max_ideological = 3.0
        
        affective_score = (
            sum(affective_scores) / len(affective_scores) / max_affective
            if affective_scores else 0.0
        )
        ideological_score = (
            sum(ideological_scores) / len(ideological_scores) / max_ideological
            if ideological_scores else 0.0
        )
        
        # Overall is weighted average
        overall_score = (affective_score + ideological_score) / 2
        
        return {
            "affective_score": affective_score,
            "ideological_score": ideological_score,
            "overall_score": overall_score
        }
    
    def _build_survey_system_prompt(self, additional_context: Optional[str] = None) -> str:
        """Build the system prompt for survey administration."""
        base_prompt = self.persona.to_system_prompt()
        survey_instructions = (
            "\n\nYou are now being asked to complete a brief survey. "
            "Please answer each question honestly based on your perspective. "
            "For each question, respond with ONLY the number that best represents "
            "your view - do not include any explanation or additional text."
        )
        
        system_prompt = base_prompt + survey_instructions
        if additional_context:
            system_prompt += f"\n\nContext: {additional_context}"
        
        return system_prompt
    
    def _build_results(
        self,
        survey_type: str,
        responses: List[SurveyResponse],
        questions: List[SurveyQuestion],
        has_conversation_context: bool = False,
        conversation_turns: int = 0
    ) -> SurveyResults:
        """Build SurveyResults from responses."""
        scores = self._calculate_scores(responses, questions)
        valid_count = sum(1 for r in responses if r.valid)
        valid_rate = valid_count / len(responses) if responses else 0.0
        
        return SurveyResults(
            survey_type=survey_type,
            responses=responses,
            affective_score=scores["affective_score"],
            ideological_score=scores["ideological_score"],
            overall_score=scores["overall_score"],
            valid_response_rate=valid_rate,
            seed=self.seed,
            metadata={
                "persona": self.persona.name,
                "provider": self.provider_class.__name__,
                "model": self.provider_kwargs.get("model_name", "unknown"),
                "temperature": self.temperature,
                "total_questions": len(questions),
                "has_conversation_context": has_conversation_context,
                "conversation_turns_in_context": conversation_turns
            }
        )
    
    def administer_pre_survey(
        self,
        additional_context: Optional[str] = None
    ) -> SurveyResults:
        """
        Administer the pre-survey with a fresh LLM instance (baseline).
        
        This creates a new LLM provider instance with NO conversation history.
        This measures the baseline attitudes before any conversation treatment.
        
        Args:
            additional_context: Optional context to add to system prompt
            
        Returns:
            SurveyResults with baseline responses and scores
        """
        logger.info(f"Administering pre-survey (baseline) with seed {self.seed}")
        
        # Create fresh provider instance
        provider = self._create_fresh_provider()
        system_prompt = self._build_survey_system_prompt(additional_context)
        
        # Administer each question (no conversation history)
        responses = []
        questions = self.survey.get_questions()
        
        for i, question in enumerate(questions):
            logger.debug(f"Pre-survey Q{i+1}/{len(questions)}: {question.id}")
            response = self._administer_single_question(
                provider, question, system_prompt, 
                conversation_history=None  # No history for pre-survey
            )
            responses.append(response)
        
        results = self._build_results(
            survey_type="pre",
            responses=responses,
            questions=questions,
            has_conversation_context=False,
            conversation_turns=0
        )
        
        logger.info(
            f"Pre-survey complete: "
            f"affective={results.affective_score:.3f}, "
            f"ideological={results.ideological_score:.3f}, "
            f"valid_rate={results.valid_response_rate:.1%}"
        )
        
        return results
    
    def administer_post_survey(
        self,
        agent: ConversationAgent,
        additional_context: Optional[str] = None
    ) -> SurveyResults:
        """
        Administer the post-survey to an agent WITH conversation history in context.
        
        This is the key method for measuring treatment effects. The conversation
        history from the agent's recent conversation remains in the context window,
        allowing measurement of how the conversation affected survey responses.
        
        Args:
            agent: The ConversationAgent that just completed the conversation.
                   Its conversation_history will be included in the context.
            additional_context: Optional additional context to add to system prompt
            
        Returns:
            SurveyResults with post-conversation responses and scores
            
        Note:
            The agent's conversation history is preserved and included with each
            survey question. This means the LLM "remembers" the conversation when
            answering the survey, which is the treatment effect we're measuring.
        """
        conversation_turns = len(agent.conversation_history)
        logger.info(
            f"Administering post-survey to agent '{agent.name}' "
            f"with {conversation_turns} turns of conversation in context"
        )
        
        # Use the agent's existing provider (same instance that had the conversation)
        provider = agent.provider
        system_prompt = self._build_survey_system_prompt(additional_context)
        
        # Get the conversation history from the agent
        conversation_history = agent.get_history()
        
        # Administer each question WITH conversation history in context
        responses = []
        questions = self.survey.get_questions()
        
        for i, question in enumerate(questions):
            logger.debug(f"Post-survey Q{i+1}/{len(questions)}: {question.id}")
            response = self._administer_single_question(
                provider, question, system_prompt,
                conversation_history=conversation_history  # Include conversation!
            )
            responses.append(response)
        
        results = self._build_results(
            survey_type="post",
            responses=responses,
            questions=questions,
            has_conversation_context=True,
            conversation_turns=conversation_turns
        )
        
        logger.info(
            f"Post-survey complete (with {conversation_turns} turns in context): "
            f"affective={results.affective_score:.3f}, "
            f"ideological={results.ideological_score:.3f}, "
            f"valid_rate={results.valid_response_rate:.1%}"
        )
        
        return results
    
    def administer_survey(
        self,
        survey_type: str = "pre",
        agent: Optional[ConversationAgent] = None,
        additional_context: Optional[str] = None
    ) -> SurveyResults:
        """
        Administer a survey (convenience method that dispatches to appropriate method).
        
        For pre-survey: Creates fresh LLM instance, no conversation context.
        For post-survey: Uses the provided agent with conversation history in context.
        
        Args:
            survey_type: "pre" or "post"
            agent: Required for post-survey - the agent with conversation history
            additional_context: Optional context to add to system prompt
            
        Returns:
            SurveyResults with responses and computed scores
            
        Raises:
            ValueError: If survey_type is "post" but no agent is provided
        """
        if survey_type == "pre":
            return self.administer_pre_survey(additional_context)
        elif survey_type == "post":
            if agent is None:
                raise ValueError(
                    "Post-survey requires an agent with conversation history. "
                    "Provide the ConversationAgent that just completed the conversation."
                )
            return self.administer_post_survey(agent, additional_context)
        else:
            raise ValueError(f"survey_type must be 'pre' or 'post', got: {survey_type}")


@dataclass
class PolarizationDelta:
    """
    Change in polarization between pre and post surveys.
    
    Positive values indicate increased polarization.
    Negative values indicate decreased polarization (depolarization).
    """
    affective_delta: float
    ideological_delta: float
    overall_delta: float
    pre_results: SurveyResults
    post_results: SurveyResults
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "affective_delta": self.affective_delta,
            "ideological_delta": self.ideological_delta,
            "overall_delta": self.overall_delta,
            "pre_survey": self.pre_results.to_dict(),
            "post_survey": self.post_results.to_dict()
        }


def calculate_polarization_delta(
    pre_results: SurveyResults,
    post_results: SurveyResults
) -> PolarizationDelta:
    """
    Calculate the change in polarization from pre to post survey.
    
    Args:
        pre_results: Results from pre-conversation survey
        post_results: Results from post-conversation survey
        
    Returns:
        PolarizationDelta with change metrics
        
    Positive delta = increased polarization (treatment effect toward polarization)
    Negative delta = decreased polarization (depolarization effect)
    """
    return PolarizationDelta(
        affective_delta=post_results.affective_score - pre_results.affective_score,
        ideological_delta=post_results.ideological_score - pre_results.ideological_score,
        overall_delta=post_results.overall_score - pre_results.overall_score,
        pre_results=pre_results,
        post_results=post_results
    )


def create_survey_experiment_protocol(
    provider_class: Type[LLMProvider],
    provider_kwargs: Dict[str, Any],
    advisor_persona: Persona,
    seed: int = 42,
    survey: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a complete experimental protocol for a pre/post survey study.
    
    This helper function sets up the survey administrator and returns
    a protocol for measuring how conversations affect LLM survey responses.
    
    The key insight: The post-survey is administered to the SAME agent that
    just had the conversation, with the conversation history still in the
    context window. This allows measuring how compressed conversations
    in context affect LLM outputs.
    
    Args:
        provider_class: LLM provider class
        provider_kwargs: Provider configuration
        advisor_persona: Persona for the advisor being surveyed
        seed: Random seed for reproducibility
        survey: Survey type to use. Options:
            - None or "default": Custom polarization survey
            - "bail2018": Bail et al. (2018) PNAS 10-item policy scale
        
    Returns:
        Dictionary with survey administrator and usage instructions
    """
    admin = SurveyAdministrator(
        provider_class=provider_class,
        provider_kwargs=provider_kwargs,
        persona=advisor_persona,
        seed=seed,
        survey=survey
    )
    
    survey_name = survey if survey else "default (PolarizationSurvey)"
    if survey and survey.lower() in ["bail2018", "bail_et_al", "bail"]:
        survey_name = "Bail et al. (2018) PNAS 10-item policy scale"
    
    return {
        "administrator": admin,
        "seed": seed,
        "survey_type": survey_name,
        "protocol": {
            "step_1": "Administer pre-survey (baseline): pre = admin.administer_pre_survey()",
            "step_2": "Create conversation agent for the advisor",
            "step_3": "Run conversation experiment (advisor talks with user)",
            "step_4": "Administer post-survey WITH conversation in context: post = admin.administer_post_survey(advisor_agent)",
            "step_5": "Calculate delta: delta = calculate_polarization_delta(pre, post)"
        },
        "notes": [
            "Pre-survey: Fresh LLM, no context (baseline measurement)",
            "Post-survey: Uses advisor agent WITH conversation history in context",
            "This measures how conversation in context window affects LLM responses",
            "Treatment = conversation with user persona (e.g., liberal/conservative)",
            "Outcome = change in survey responses (delta)"
        ]
    }


def list_available_surveys() -> Dict[str, str]:
    """
    List all available built-in survey types.
    
    Returns:
        Dictionary mapping survey type keys to descriptions
        
    Example:
        >>> from synthetic_experiments.analysis.survey import list_available_surveys
        >>> surveys = list_available_surveys()
        >>> for key, desc in surveys.items():
        ...     print(f"{key}: {desc}")
    """
    return {
        "default": (
            "PolarizationSurvey - Custom survey with 6 affective polarization "
            "questions (feeling thermometers, trust, social distance) and "
            "6 ideological questions (climate, healthcare, immigration, taxes, "
            "guns, certainty)"
        ),
        "bail2018": (
            "BailEtAlSurvey - Replicates Bail et al. (2018) PNAS study. "
            "10-item liberalism-conservatism policy scale (α=.91). "
            "Covers government regulation, welfare, immigration, race, "
            "corporations, homosexuality, and military strength. "
            "Higher scores = more conservative positions."
        ),
        "bail_et_al": "Alias for 'bail2018'",
        "bail": "Alias for 'bail2018'",
    }
