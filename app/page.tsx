import Accordion from "../components/Accordion";
import accordionData from "../data/accordionData";

export default function Home() {
  return (
    <div>
      <h1>Accordion Example</h1>
      <Accordion data={accordionData} />
    </div>
  );
}