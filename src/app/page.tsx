import React from 'react';
import { LayerView } from '@/src/llm/LayerView';
import { InfoButton } from '@/src/llm/WelcomePopup';
import { Header } from '@/src/homepage/Header';

export const metadata = {
  title: 'LLM Visualization',
  description: 'A 3D animated visualization of an LLM with a walkthrough.',
};

export default function Page() {
    return <>
        <Header title={<span className="flex items-baseline gap-2">
            <span>LLM Visualization</span>
            <span className="text-xs font-normal text-blue-200 whitespace-nowrap">
                改编自 <a href="https://bbycroft.net/llm" target="_blank" rel="noreferrer"
                   className="underline hover:text-white">Brendan Bycroft 的 llm-viz</a>
            </span>
        </span>}>
            <InfoButton />
        </Header>
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
