import { Calendar as LucideCalendar, LucideProps } from 'lucide-react';

const CalenderLine = ({ className, ...props }: LucideProps) => {
  return <LucideCalendar className={className} {...props} />;
};

export default CalenderLine;