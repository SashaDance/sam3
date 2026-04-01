
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(checkpoint_path="/workspace/sam3/sam3.pt")
video_path = 'data/indoor_example.mp4'
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="chair",
    )
)
output = response["outputs"]
print(output)
