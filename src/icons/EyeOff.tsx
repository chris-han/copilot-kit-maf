import * as React from "react";

const EyeOff = ({ className, ...props }: React.SVGProps<SVGSVGElement>) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    {...props}
  >
    <path d="M9.309 4.791A1 1 0 0 1 10.06 4.2a10.751 10.751 0 0 1 11.66 11.66 1 1 0 0 1-.596.747M5.937 7.057A10.75 10.75 0 0 0 2.062 12a10.75 10.75 0 0 0 19.876 0 8.5 8.5 0 0 0-4.934-5.895M12 16a4 4 0 1 1 0-8 4 4 0 0 1 0 8" />
    <path d="M4.629 4.629 4 4" />
    <path d="m19.98 4.02-1.56 1.56" />
  </svg>
);

export default EyeOff;