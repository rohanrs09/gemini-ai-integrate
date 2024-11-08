import Accordion from "../components/Accordion";
import accordionData from "../data/accordionData";

export default function Home() {
  return (
    <div>
      <div className="flex flex-col items-center justify-center ">
  <h1 className="text-2xl font-bold mb-4">Accordion Example</h1>
  <a
    href="https://github.com/b1ink0/SPPU-BE-IT-DL-LP-IV"
    target="_blank"
    rel="noopener noreferrer"
    className="text-blue-600 underline"
  >
    GitHub
  </a>
</div>

      <Accordion data={accordionData} />
    </div>
  );
}
