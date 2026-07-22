import { RoadmapPage, collectLinkedUrls } from '@freemocap/skellydocs';
import config from '../../content.config';

const REPO = 'freemocap/skellytracker';

export default function Roadmap() {
  return (
    <RoadmapPage
      repo={REPO}
      pinnedIssues={collectLinkedUrls(config)}
      projectBoardUrl={config.projectBoardUrl}
    />
  );
}
