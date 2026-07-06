# Eval: the video-generation king-of-the-hill duel (scoring + verdict).
#
# The heavy generation stack (torch / diffusers) is imported lazily inside
# `video_runner` so the pure scoring logic here (`bootstrap`) stays import-safe
# and unit-testable without a GPU.
