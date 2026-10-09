#pragma once
void *runtime_workspace_state(void);
void runtime_workspace_bind(void *space);
void runtime_workspace_register(void *space);
void runtime_workspace_check_idle(void);
