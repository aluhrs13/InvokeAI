/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

/**
 * Finds a dynamic prompt
 */
export type DynamicPromptInvocation = {
    /**
     * The id of this node. Must be unique among all nodes.
     */
    id: string;
    type?: 'dynamic_prompt';
    /**
     * The first number
     */
    'prompt'?: string;
  };
  
  