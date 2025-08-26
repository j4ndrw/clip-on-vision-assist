from http import HTTPStatus
from flask import Response, json, request
from flask_pydantic import validate

from src.control_center.models.llm.endpoint.amend_llm_configuration import AmendLLMConfigurationRequest
from src.control_center.models.llm.endpoint.get_available_llms import GetAvailableLLMsResponse
from src.control_center.models.llm.endpoint.get_current_llm_configuration import GetCurrentLLMConfigurationResponse
from src.control_center.models.llm.endpoint.get_llm_endpoint_suggestions import GetLLMEndpointSuggestionsResponse
from src.control_center.services.llm.get_available_llms import get_available_llms
from src.control_center.services.llm.get_current_llm_configuration import get_current_llm_configuration
from src.control_center.services.llm.amend_llm_configuration import amend_llm_configuration
from src.control_center.services.llm.get_llm_endpoint_suggestions import get_llm_endpoint_suggestions

from flask import Blueprint

bp = Blueprint('llm', __name__)

@bp.get("/config")
@validate()
def get_current_llm_configuration_route():
    llm_config = get_current_llm_configuration()
    return GetCurrentLLMConfigurationResponse(llm_config=llm_config).as_json()

@bp.post("/config")
@validate(body=AmendLLMConfigurationRequest, response_by_alias=True)
def amend_llm_configuration_route():
    body = AmendLLMConfigurationRequest(**request.json or {})
    err = amend_llm_configuration(endpoint=body.endpoint, model=body.model, api_key=body.api_key)
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(response=json.dumps({}), status=HTTPStatus.OK)

@bp.get("/list")
@validate()
def get_available_llms_route():
    llms, err = get_available_llms()
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)

    return GetAvailableLLMsResponse(llms=llms).as_json()

@bp.get("/endpoint-suggestions")
@validate()
def get_llm_endpoint_suggestions_route():
    endpoint_suggestions = get_llm_endpoint_suggestions()
    return GetLLMEndpointSuggestionsResponse(endpoint_suggestions=endpoint_suggestions).as_json()
